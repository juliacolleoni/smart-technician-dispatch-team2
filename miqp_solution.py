
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import cvxpy as cp
import argparse

def get_technicians_df():
    technicians_df = pd.read_csv('01 - Smart Technician Dispatch/01_technician_profiles.csv')
    return technicians_df

def get_workorders_df(day):
    workorders_df = pd.read_csv('01 - Smart Technician Dispatch/04_workorders_week_original.csv')
    return workorders_df[workorders_df['scheduled_date'] == day].copy()

def get_calendar_df():
    calendar_df = pd.read_csv('01 - Smart Technician Dispatch/05_technician_calendar_original.csv')
    return calendar_df

def duration_to_slots(duration_minutes, slot_size_minutes=15):
    """
    Convert duration in minutes to number of time slots.
    Rounds up to nearest slot (31 minutes becomes 3 slots = 45 minutes with 15min slots).
    
    Args:
        duration_minutes: Duration in minutes
        slot_size_minutes: Size of each time slot (default 15 minutes)
    
    Returns:
        Number of slots needed
    """
    return math.ceil(duration_minutes / slot_size_minutes)

def time_to_slot(time_str, shift_start_hour=8, slot_size_minutes=15):
    """
    Convert time string (e.g., "10:00" or "10:00:00") to slot number.
    
    Args:
        time_str: Time string in format "HH:MM" or "HH:MM:SS"
        shift_start_hour: Hour when shift starts (default 8 for 08:00)
        slot_size_minutes: Size of each time slot (default 15 minutes)
    
    Returns:
        Slot number (0-indexed)
    """
    time_parts = time_str.split(':')
    hour = int(time_parts[0])
    minute = int(time_parts[1])
    total_minutes = (hour - shift_start_hour) * 60 + minute
    return total_minutes // slot_size_minutes

def parse_preferred_window(window_str, slot_size_minutes=15):
    """
    Parse customer preferred window and return start/end slot numbers.
    
    Args:
        window_str: Time window string like "08:00-10:00"
        slot_size_minutes: Size of each time slot (default 15 minutes)
    
    Returns:
        tuple: (start_slot, end_slot) - both inclusive
    """
    start_time, end_time = window_str.split('-')
    start_slot = time_to_slot(start_time, slot_size_minutes=slot_size_minutes)
    end_slot = time_to_slot(end_time, slot_size_minutes=slot_size_minutes)
    return start_slot, end_slot

def latlon_to_xyz(lat, lon, radius=6371.0):
    """
    Convert latitude/longitude to 3D Cartesian coordinates (x, y, z).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        radius: Earth radius in km (default 6371.0)
    
    Returns:
        tuple: (x, y, z) coordinates in km
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z

def euclidean_distance_3d(x1, y1, z1, x2, y2, z2):
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        x1, y1, z1: First point coordinates
        x2, y2, z2: Second point coordinates
    
    Returns:
        float: Euclidean distance
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='MIQP Technician Dispatch Optimization')
parser.add_argument('--output', '-o', type=str, default='final_schedule.csv',
                    help='Output CSV filename (default: final_schedule.csv)')
args = parser.parse_args()

# Configuration
shift_start = '08:00:00'
shift_end = '18:00:00'

# Get all unique days from workorders
all_workorders_df = pd.read_csv('01 - Smart Technician Dispatch/04_workorders_week_original.csv')
all_days = sorted(all_workorders_df['scheduled_date'].unique())

print(f"Found {len(all_days)} days to process: {all_days}")
print(f"Output file: {args.output}")

# Load technicians (same for all days)
workers = get_technicians_df()

# Load calendar (filter per day later)
calendar_df = get_calendar_df()

# Calculate time slots per day
# From 08:00 to 18:00 = 10 hours = 600 minutes = 40 slots of 15 minutes
slot_size_minutes = 15
N = int((18 - 8) * 60 / slot_size_minutes)  # 40 time slots

# Number of workers (constant across all days)
M = len(workers)

# Worker home base locations (depot) - constant across all days
worker_home_xyz = []
for m in range(M):
    lat = workers.iloc[m]['home_base_lat']
    lon = workers.iloc[m]['home_base_lon']
    x, y, z = latlon_to_xyz(lat, lon)
    worker_home_xyz.append((x, y, z))

def calculate_skill_match(workers, orders):
    """Calculate skills match matrix for given workers and orders."""
    M = len(workers)
    K = len(orders)
    skill_match = np.zeros((M, K))
    
    for m in range(M):
        # Parse worker skills (comma-separated string)
        worker_skills_str = workers.iloc[m]['skills']
        if pd.isna(worker_skills_str) or worker_skills_str == '':
            worker_skills = set()
        else:
            worker_skills = set(skill.strip() for skill in str(worker_skills_str).split(','))
        
        for k in range(K):
            # Parse order required skills (comma-separated string)
            order_skills_str = orders.iloc[k]['required_skills_ground_truth']
            if pd.isna(order_skills_str) or order_skills_str == '':
                required_skills = set()
            else:
                required_skills = set(skill.strip() for skill in str(order_skills_str).split(','))
            
            # Calculate match ratio
            if len(required_skills) == 0:
                # If no skills required, any worker is a perfect match
                skill_match[m, k] = 1.0
            else:
                # Calculate fraction of required skills that worker has
                matched_skills = worker_skills.intersection(required_skills)
                skill_match[m, k] = len(matched_skills) / len(required_skills)
    
    return skill_match

def calculate_max_coord_diff(order_coords_xyz, worker_home_xyz):
    """Calculate maximum coordinate difference for Big-M."""
    all_coords_xyz = order_coords_xyz + worker_home_xyz
    all_x = [c[0] for c in all_coords_xyz]
    all_y = [c[1] for c in all_coords_xyz]
    all_z = [c[2] for c in all_coords_xyz]
    
    max_x_diff = max(all_x) - min(all_x)
    max_y_diff = max(all_y) - min(all_y)
    max_z_diff = max(all_z) - min(all_z)
    max_coord_diff = max(max_x_diff, max_y_diff, max_z_diff)
    
    return max_coord_diff

def get_technician_availability(calendar_df, workers_df, day, N=40, slot_size_minutes=15):
    """
    Parse technician calendar to determine which time slots are unavailable.
    
    Returns:
        dict: {technician_id: set of unavailable slot indices}
              Unavailable slots include: BREAK, MEETING, ADMIN, UNAVAILABLE
    """
    # Filter calendar for the specific day
    day_calendar = calendar_df[calendar_df['date'] == day].copy()
    
    # Event types that block scheduling
    blocking_events = ['BREAK', 'MEETING', 'ADMIN', 'UNAVAILABLE']
    
    unavailable_slots = {}
    
    for m in range(len(workers_df)):
        tech_id = workers_df.iloc[m]['technician_id']
        tech_events = day_calendar[day_calendar['technician_id'] == tech_id]
        
        blocked_slots = set()
        
        for _, event in tech_events.iterrows():
            if event['event_type'] in blocking_events:
                # Parse start and end times
                if pd.notna(event['start_time']) and pd.notna(event['end_time']):
                    start_slot = time_to_slot(event['start_time'], slot_size_minutes=slot_size_minutes)
                    end_slot = time_to_slot(event['end_time'], slot_size_minutes=slot_size_minutes)
                    
                    # Add all slots in this time range to blocked list
                    for slot in range(start_slot, end_slot):
                        if 0 <= slot < N:
                            blocked_slots.add(slot)
        
        unavailable_slots[tech_id] = blocked_slots
    
    return unavailable_slots

def distance_score(y, p, order_coords_xyz, worker_home_xyz, max_coord_diff, M, N, K):
    """
    Constraints and objective for minimizing travel distance using Big-M formulation.
    
    Args:
        y: Binary assignment variables
        p: Position variables
        order_coords_xyz: List of order coordinates
        worker_home_xyz: List of worker home coordinates
        max_coord_diff: Maximum coordinate difference for Big-M
        M: Number of workers
        N: Number of time slots
        K: Number of orders
    
    Position dynamics:
    - When y[i,m,k]=1 (job k starts), position moves to job k's location
    - When no job starts, position stays the same
    
    Returns:
        tuple: (constraints, objective_cost)
            - constraints: list of cvxpy constraints for position tracking
            - objective_cost: cvxpy expression for total travel distance (sum of squared distances)
    """
    constraints = []
    
    # Big-M constant: upper bound on distance between any two points
    # Use 2x the actual maximum coordinate difference for safety margin
    M_big = max_coord_diff * 2.0
    
    # Initial positions: All workers start at their home base (depot)
    for m in range(M):
        x_home, y_home, z_home = worker_home_xyz[m]
        constraints += [
            p[0, m, 0] == x_home,
            p[0, m, 1] == y_home,
            p[0, m, 2] == z_home
        ]
    
    # Position dynamics using Big-M formulation
    # At each time slot, position either:
    # 1) Stays the same (if no job starts), OR
    # 2) Moves to a job location (if a job starts)
    
    for m in range(M):
        for i in range(N):
            # Count total jobs starting at time i for worker m
            total_jobs_starting = cp.sum([y[i, m, k] for k in range(K)])
            
            for coord_idx in range(3):  # x, y, z coordinates
                # Case 1: If a job k starts at time i, position moves to that job's location
                # For each job k: if y[i,m,k]=1, then p[i+1,m,:] = order_coords_xyz[k]
                for k in range(K):
                    job_coord = order_coords_xyz[k][coord_idx]
                    # Big-M: |p[i+1] - job_coord| <= M_big * (1 - y[i,m,k])
                    # If y[i,m,k]=1, then p[i+1] must equal job_coord
                    # If y[i,m,k]=0, constraint is always satisfied
                    constraints += [
                        p[i+1, m, coord_idx] - job_coord <= M_big * (1 - y[i, m, k]),
                        p[i+1, m, coord_idx] - job_coord >= -M_big * (1 - y[i, m, k])
                    ]
                
                # Case 2: If no job starts, position stays the same
                # |p[i+1] - p[i]| <= M_big * total_jobs_starting
                # If total_jobs_starting=0, then p[i+1] must equal p[i]
                # If total_jobs_starting>=1, constraint is always satisfied

                constraints += [
                    p[i+1, m, coord_idx] - p[i, m, coord_idx] <= M_big * total_jobs_starting,
                    p[i+1, m, coord_idx] - p[i, m, coord_idx] >= -M_big * total_jobs_starting
                ]
    
    # Calculate total travel distance (sum of squared distances between consecutive positions)
    # Using squared distance to keep the objective convex (avoid sqrt)
    total_distance_squared = 0
    for m in range(M):
        for i in range(N):
            # Distance from position at time i to position at time i+1
            dx = p[i+1, m, 0] - p[i, m, 0]
            dy = p[i+1, m, 1] - p[i, m, 1]
            dz = p[i+1, m, 2] - p[i, m, 2]
            
            distance_squared = dx**2 + dy**2 + dz**2
            total_distance_squared += distance_squared
    
    distance_cost = total_distance_squared
    
    return constraints, distance_cost


def skills_score():
    """
    Objective for maximizing skills match between workers and orders.
    
    Uses pre-calculated skill_match[m,k] matrix where each entry represents
    the fraction of required skills that worker m possesses for order k.
    
    Objective maximizes the total skill match quality across all assignments.
    Since we're minimizing in the main objective, we return negative cost.
    
    Returns:
        tuple: (constraints, objective_cost)
            - constraints: empty list (no additional constraints)
            - objective_cost: negative sum of skill matches (to be minimized)
    """
    constraints = []
    
    # Calculate total skills mismatch across all assignments
    # For each assignment y[i,m,k]=1, we want to reward skill_match[m,k]
    # Since we minimize, we use (1 - skill_match) as the cost
    # This way: perfect match (1.0) has cost 0, no match (0.0) has cost 1
    total_mismatch = 0
    
    for m in range(M):
        for k in range(K):
            # Cost for assigning worker m to order k
            # mismatch = 1 - skill_match (0 is best, 1 is worst)
            mismatch = 1.0 - skill_match[m, k]
            
            # Sum over all time slots where this assignment could start
            # If y[i,m,k]=1, we incur the mismatch cost
            for i in range(N):
                total_mismatch += mismatch * y[i, m, k]
    
    skills_cost = total_mismatch
    
    return constraints, skills_cost


def calendar_score():
    """
    Constraints and objective related to customer preferred time windows.
    
    Uses a single slack variable per order to measure deviation from the preferred window.
    The slack s[k] represents:
    - 0 if the order starts within the preferred window
    - (pref_start - actual_start) if the order starts too early
    - (actual_start - pref_end) if the order starts too late
    
    Returns:
        tuple: (constraints, objective_cost)
            - constraints: list of cvxpy constraints for preference windows
            - objective_cost: cvxpy expression for preference violation penalty
    """
    constraints = []
    
    # Slack variable constraints to measure deviation from preferred time windows
    for k in range(K):
        pref_start, pref_end = preferred_windows[k]
        
        # Calculate actual start time: sum over all possible start times weighted by y
        # actual_start = sum_{i,m} i * y[i,m,k]
        actual_start = cp.sum([i * y[i, m, k] for i in range(N) for m in range(M)])
        
        # The slack s[k] must be large enough to cover deviation on either side:
        # s[k] >= pref_start - actual_start  (if starting early)
        # s[k] >= actual_start - pref_end    (if starting late)
        # s[k] >= 0                          (if within window)
        #
        # Since s[k] is minimized in the objective and is non-negative,
        # it will equal max(0, pref_start - actual_start, actual_start - pref_end)
        constraints += [s[k] >= pref_start - actual_start]
        constraints += [s[k] >= actual_start - pref_end]
    
    # Preference violation cost: sum of all slack variables
    preference_cost = cp.sum(s)
    
    return constraints, preference_cost


def core_opt(max_jobs_per_tech, unavailable_slots):
    """
    Core optimization constraints for technician-order assignment.
    
    Args:
        max_jobs_per_tech: dict mapping technician_id to max jobs per day
        unavailable_slots: dict mapping technician_id to set of blocked time slots
    
    Returns:
        list: cvxpy constraints for the core assignment problem
    """
    constraints = []
    constraints += [0 <= x, x <= 1, 0 <= y, y <= 1]

    # Prevent orders from starting too late (they wouldn't fit before end of time horizon)
    for k in range(K):
        for m in range(M):
            for i in range(N):
                # If order k has duration W[k] and starts at time i, 
                # it needs slots i through i+W[k]-1
                # So it can only start if i + W[k] <= N
                if i + W[k] > N:
                    constraints += [y[i, m, k] == 0]
    
    # CONSTRAINT 1: Max jobs per day per technician
    for m in range(M):
        tech_id = workers.iloc[m]['technician_id']
        max_jobs = max_jobs_per_tech.get(tech_id, 999)  # Default to high number if not specified
        
        # Total jobs assigned to this technician cannot exceed max_jobs
        total_jobs_assigned = cp.sum([y[i, m, k] for i in range(N) for k in range(K)])
        constraints += [total_jobs_assigned <= max_jobs]
    
    # CONSTRAINT 2: Technician availability (block unavailable time slots)
    for m in range(M):
        tech_id = workers.iloc[m]['technician_id']
        blocked_slots = unavailable_slots.get(tech_id, set())
        
        # For each blocked slot, no job can start that would occupy that slot
        for blocked_slot in blocked_slots:
            for k in range(K):
                # Find all start times i where job k would occupy blocked_slot
                # Job starting at i occupies slots [i, i+1, ..., i+W[k]-1]
                # So it occupies blocked_slot if: i <= blocked_slot < i + W[k]
                # Which means: blocked_slot - W[k] + 1 <= i <= blocked_slot
                
                for i in range(max(0, blocked_slot - W[k] + 1), min(N, blocked_slot + 1)):
                    # If job k starts at time i, it occupies slot range [i, i+W[k])
                    if i <= blocked_slot < i + W[k]:
                        constraints += [y[i, m, k] == 0]

    for m in range(M): 
        limit_constraints = [0] * N
        for i in range(N):
            for j in range(max(W)):
                # Skip if this would go out of bounds
                if i + j >= N: 
                    continue
                    
                g = 0
                
                for k in range(K):
                    w = W[k]
                    if j < w:
                        g += y[i][m][k]


                constraints += [
                    x[i+j][m] >= g
                ]

                limit_constraints[i+j] += g
            
        for i in range(N):
            constraints += [
                limit_constraints[i] <= 1
            ]
            constraints += [
                x[i][m] <= limit_constraints[i]
            ]

    for k in range(K):
        constraints += [cp.sum(y[:, :, k]) == 1]
    
    return constraints


# Process all days sequentially
all_final_schedules = []

for day_idx, day in enumerate(all_days):
    print(f"\n{'='*80}")
    print(f"PROCESSING DAY {day_idx+1}/{len(all_days)}: {day}")
    print(f"{'='*80}")
    
    # Get workorders for this day
    orders = get_workorders_df(day)
    K = len(orders)
    print(f"Number of orders: {K}")
    
    if K == 0:
        print(f"No orders for {day}, skipping...")
        continue

    # Convert workorder durations to slots
    W = [duration_to_slots(orders.iloc[k]['job_duration_minutes']) for k in range(K)]
    
    # Parse preferred time windows
    preferred_windows = [parse_preferred_window(orders.iloc[k]['customer_preferred_window']) for k in range(K)]
    
    # Convert order locations to XYZ coordinates
    order_coords_xyz = []
    for k in range(K):
        lat = orders.iloc[k]['job_lat']
        lon = orders.iloc[k]['job_lon']
        x, y, z = latlon_to_xyz(lat, lon)
        order_coords_xyz.append((x, y, z))
    
    # Calculate maximum coordinate difference for Big-M
    max_coord_diff = calculate_max_coord_diff(order_coords_xyz, worker_home_xyz)
    print(f"Max coordinate difference: {max_coord_diff:.2f} km")
    print(f"Setting M_big = {max_coord_diff * 2.0:.2f} km (2x safety margin)")
    
    # Calculate skills match matrix
    skill_match = calculate_skill_match(workers, orders)
    print(f"\nSkills matching statistics:")
    print(f"  Average skill match: {skill_match.mean():.2f}")
    print(f"  Perfect matches available: {(skill_match == 1.0).sum()} out of {M*K}")
    
    # Create optimization variables
    print(f"\nCreating optimization variables...")
    y = cp.Variable((N, M, K), boolean=True)  # y[i,m,k] = 1 if worker m starts order k at time i
    x = cp.Variable((N, M), boolean=True)     # x[i,m] = 1 if worker m is busy at time i
    s = cp.Variable(K, nonneg=True)           # s[k] = slack for preference window violation of order k
    p = cp.Variable((N+1, M, 3))              # p[i,m,coord] = position of worker m at time i (x,y,z)
    
    print(f"Variables created:")
    print(f"  y: {y.shape} (assignment variables)")
    print(f"  x: {x.shape} (busy variables)")
    print(f"  s: {s.shape} (slack variables)")
    print(f"  p: {p.shape} (position variables)")
    
    # Get max jobs per day for each technician
    max_jobs_per_tech = {workers.iloc[m]['technician_id']: workers.iloc[m]['max_jobs_per_day'] for m in range(M)}
    
    # Get technician availability (blocked time slots)
    unavailable_slots = get_technician_availability(calendar_df, workers, day, N, slot_size_minutes)
    
    # Print availability summary
    print(f"\nTechnician availability:")
    for tech_id, blocked in unavailable_slots.items():
        if blocked:
            print(f"  {tech_id}: {len(blocked)} blocked slots - {sorted(list(blocked))}")
    
    # Build the complete optimization problem
    core_constraints = core_opt(max_jobs_per_tech, unavailable_slots)
    calendar_constraints, preference_cost = calendar_score()
    skills_constraints, skills_cost = skills_score()
    distance_constraints, distance_cost = distance_score(y, p, order_coords_xyz, worker_home_xyz, max_coord_diff, M, N, K)
    
    # Combine all constraints
    all_constraints = core_constraints + calendar_constraints + skills_constraints + distance_constraints
    
    # Objective: Minimize multiple objectives with weights
    preference_weight = 10.0
    skills_weight = 5.0
    distance_weight = 0.00001  # Small weight for distance (in squared km)
    
    objective = cp.Minimize(
        preference_weight * preference_cost +
        skills_weight * skills_cost +
        distance_weight * distance_cost
    )
    
    problem = cp.Problem(objective, all_constraints)
    
    solver_options = {
        'verbose': False,  # Less verbose for multi-day
        'warm_start': False,  # Disabled: SCIP cannot handle warm start with distance optimization (aggregated variables)
    }
    
    # Solve
    print("\nSolving optimization...")
    result = problem.solve(solver=cp.SCIP, **solver_options)
    
    print(f'\nOptimization Status: {problem.status}')
    print(f'Objective value: {result:.4f}')
    
    def slot_to_time(slot_num, start_hour=8, slot_minutes=30):
        """Convert slot number to time string"""
        total_minutes = start_hour * 60 + slot_num * slot_minutes
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    
    if problem.status == 'optimal':
        assignments = []
        for k in range(K):
            for m in range(M):
                for i in range(N):
                    if y.value[i, m, k] > 0.5:  # Assignment found
                        workorder_id = orders.iloc[k]['workorder_id']
                        tech_id = workers.iloc[m]['technician_id']
                        tech_name = workers.iloc[m]['technician_name']
                        start_time = slot_to_time(i)
                        end_slot = i + W[k]
                        end_time = slot_to_time(end_slot)
                        duration = orders.iloc[k]['job_duration_minutes']
                        
                        # Calculate preference violation
                        pref_start, pref_end = preferred_windows[k]
                        pref_window = orders.iloc[k]['customer_preferred_window']
                        early_violation = max(0, pref_start - i)
                        late_violation = max(0, i - pref_end)
                        in_window = (early_violation == 0 and late_violation == 0)
                        
                        assignments.append({
                            'workorder_id': workorder_id,
                            'order_index': k,
                            'technician_id': tech_id,
                            'technician_name': tech_name,
                            'start_slot': i,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration_min': duration,
                            'slots': W[k],
                            'preferred_window': pref_window,
                            'pref_start_slot': pref_start,
                            'pref_end_slot': pref_end,
                            'early_by_slots': early_violation,
                            'late_by_slots': late_violation,
                            'in_preferred_window': in_window
                        })
        
        # Sort by technician and start time
        assignments.sort(key=lambda x: (x['technician_id'], x['start_slot']))
        
        # Calculate worker distances for travel scores
        worker_distances = {}
        for m in range(M):
            tech_id = workers.iloc[m]['technician_id']
            worker_distance = 0
            for i in range(N):
                dx = p.value[i+1, m, 0] - p.value[i, m, 0]
                dy = p.value[i+1, m, 1] - p.value[i, m, 1]
                dz = p.value[i+1, m, 2] - p.value[i, m, 2]
                worker_distance += np.sqrt(dx**2 + dy**2 + dz**2)
            worker_distances[tech_id] = worker_distance
        
        # Convert to final_schedule format
        from convert_miqp_to_final_schedule import convert_assignments_to_final_schedule
        
        day_schedule = convert_assignments_to_final_schedule(
            assignments=assignments,
            orders_df=orders,
            workers_df=workers,
            skill_match=skill_match,
            preferred_windows=preferred_windows,
            worker_distances=worker_distances,
            scheduled_date=day,
            N=N
        )
        
        all_final_schedules.append(day_schedule)
        
        # Print summary
        in_window = sum(1 for a in assignments if a['in_preferred_window'])
        total_dist = sum(worker_distances.values())
        print(f"  Assigned {len(assignments)}/{K} orders")
        print(f"  In preferred window: {in_window}/{K} ({100*in_window/K:.1f}%)")
        print(f"  Total distance: {total_dist:.2f} km")
        print(f"  Distance cost: {distance_cost.value:.2f}")
    
    else:
        print(f"  Optimization failed: {problem.status}")
        continue  # Skip to next day

# Combine all days into final_schedule.csv
if all_final_schedules:
    print(f"\n{'='*80}")
    print(f"COMBINING RESULTS FROM ALL DAYS")
    print(f"{'='*80}")
    
    final_schedule_df = pd.concat(all_final_schedules, ignore_index=True)
    
    # Save to CSV
    final_schedule_df.to_csv(args.output, index=False)
    
    print(f"✓ Saved {len(final_schedule_df)} total assignments to {args.output}")
    print(f"\nBreakdown by day:")
    
    for day, schedule_df in zip(all_days, all_final_schedules):
        print(f"  {day}: {len(schedule_df)} assignments")
    
    print(f"\nSample output (first 10 rows):")
    print(final_schedule_df.head(10).to_string())
else:
    print("\nNo successful optimizations to save.")

print(f"\n{'='*80}")
print(f"OPTIMIZATION COMPLETE")
print(f"{'='*80}")
