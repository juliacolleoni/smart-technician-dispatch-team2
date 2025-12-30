"""
Smart Technician Dispatch Optimizer
TELUS CAIO B2B Hackathon #1
December 29-31, 2025

This solution optimizes technician scheduling by balancing:
1. Skill matching
2. Calendar availability
3. Route optimization (travel distance)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Set
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Scoring weights (must sum to 1.0)
SKILL_WEIGHT = 0.35
AVAILABILITY_WEIGHT = 0.30
TRAVEL_WEIGHT = 0.35

# Constraint enforcement mode
# 'strict': Zero violations, may leave some jobs unassigned
# 'flexible': Allow minor violations (+1 job) to achieve 100% assignment
CONSTRAINT_MODE = 'flexible'  # Change to 'strict' for zero violations

# Common technical skills to extract from job descriptions
SKILL_KEYWORDS = {
    'fiber': ['fiber', 'fibre', 'ftth', 'ont'],
    'internet': ['internet', 'broadband', 'wifi', 'wi-fi', 'router', 'modem'],
    'tv': ['tv', 'television', 'optik', 'set-top', 'settop', 'stb'],
    'phone': ['phone', 'voip', 'landline', 'telephone'],
    'install': ['install', 'installation', 'setup', 'new service'],
    'repair': ['repair', 'fix', 'broken', 'issue', 'problem'],
    'troubleshoot': ['troubleshoot', 'diagnostic', 'diagnose', 'investigate'],
    'upgrade': ['upgrade', 'enhance', 'improve'],
    'network': ['network', 'connectivity', 'connection'],
    'hardware': ['hardware', 'device', 'equipment'],
    'cable': ['cable', 'cabling', 'wiring'],
    'config': ['config', 'configuration', 'settings'],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using haversine formula.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def time_to_minutes(time_str) -> int:
    """Convert time string (HH:MM) or time object to minutes since midnight."""
    if pd.isna(time_str) or time_str == '':
        return 0
    try:
        # Handle datetime.time objects
        if hasattr(time_str, 'hour'):
            return time_str.hour * 60 + time_str.minute
        # Handle string format
        time_str = str(time_str)
        if ':' in time_str:
            parts = time_str.split(':')
            h, m = int(parts[0]), int(parts[1])
            return h * 60 + m
        return 0
    except:
        return 0

def minutes_to_time(minutes: int) -> str:
    """Convert minutes since midnight to HH:MM format."""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

# ============================================================================
# SKILL EXTRACTION
# ============================================================================

class SkillExtractor:
    """Extract required skills from work order descriptions."""
    
    def __init__(self):
        self.skill_keywords = SKILL_KEYWORDS
    
    def extract_skills(self, description: str) -> Set[str]:
        """Extract skills from a job description using NLP (fallback method)."""
        if pd.isna(description):
            return set()
        
        description_lower = str(description).lower()
        extracted_skills = set()
        
        for skill, keywords in self.skill_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    extracted_skills.add(skill)
                    break
        
        return extracted_skills
    
    def parse_ground_truth_skills(self, ground_truth: str) -> Set[str]:
        """Parse ground truth skills from semicolon-separated string."""
        if pd.isna(ground_truth) or ground_truth == '':
            return set()
        
        # Split by semicolon and clean up
        skills = str(ground_truth).lower().split(';')
        skills = {s.strip() for s in skills if s.strip()}
        
        return skills
    
    def calculate_skill_match_score(self, required_skills: Set[str], 
                                    technician_skills: str) -> float:
        """
        Calculate skill match score between required and technician skills.
        Returns score between 0 and 1.
        """
        if not required_skills:
            return 0.5  # Neutral score if no specific skills identified
        
        if pd.isna(technician_skills):
            return 0.0
        
        # Parse technician skills (semicolon-separated)
        tech_skills = set(str(technician_skills).lower().split(';'))
        tech_skills = {s.strip() for s in tech_skills}
        
        # Count matching skills
        matches = 0
        for req_skill in required_skills:
            for tech_skill in tech_skills:
                # Check for exact match or substring match
                if req_skill == tech_skill or req_skill in tech_skill or tech_skill in req_skill:
                    matches += 1
                    break
        
        # Score based on match ratio
        match_ratio = matches / len(required_skills) if required_skills else 0
        return match_ratio

# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Load and preprocess all datasets."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.technicians = None
        self.customers = None
        self.service_history = None
        self.workorders = None
        self.calendar = None
        self.locations = None
        
    def load_all(self):
        """Load all datasets."""
        print("Loading datasets...")
        
        self.technicians = pd.read_excel(f"{self.data_dir}/01_technician_profiles.xlsx")
        self.customers = pd.read_excel(f"{self.data_dir}/02_customer_profiles.xlsx")
        self.service_history = pd.read_excel(f"{self.data_dir}/03_customer_service_history.xlsx")
        self.workorders = pd.read_excel(f"{self.data_dir}/04_workorders_week_original.xlsx")
        self.calendar = pd.read_excel(f"{self.data_dir}/05_technician_calendar_original.xlsx")
        self.locations = pd.read_excel(f"{self.data_dir}/06_locations_nodes.xlsx")
        
        print(f"✓ Loaded {len(self.technicians)} technicians")
        print(f"✓ Loaded {len(self.customers)} customers")
        print(f"✓ Loaded {len(self.service_history)} service history records")
        print(f"✓ Loaded {len(self.workorders)} work orders")
        print(f"✓ Loaded {len(self.calendar)} calendar entries")
        print(f"✓ Loaded {len(self.locations)} location nodes")
        
        return self

# ============================================================================
# DISTANCE CALCULATOR
# ============================================================================

class DistanceCalculator:
    """Calculate and cache distances between locations."""
    
    def __init__(self, locations_df: pd.DataFrame):
        self.locations = locations_df
        self.distance_cache = {}
        self._build_distance_matrix()
    
    def _build_distance_matrix(self):
        """Pre-calculate all pairwise distances."""
        print("\nBuilding distance matrix...")
        
        for i, loc1 in self.locations.iterrows():
            for j, loc2 in self.locations.iterrows():
                if i != j:
                    key = (loc1['node_id'], loc2['node_id'])
                    if key not in self.distance_cache:
                        dist = haversine_distance(
                            loc1['lat'], loc1['lon'],
                            loc2['lat'], loc2['lon']
                        )
                        self.distance_cache[key] = dist
                        self.distance_cache[(loc2['node_id'], loc1['node_id'])] = dist
        
        print(f"✓ Calculated {len(self.distance_cache)} distances")
    
    def get_distance(self, loc1_id: str, loc2_id: str) -> float:
        """Get distance between two locations."""
        return self.distance_cache.get((loc1_id, loc2_id), 0.0)

# ============================================================================
# AVAILABILITY MANAGER
# ============================================================================

class AvailabilityManager:
    """Manage technician availability and schedule conflicts."""
    
    def __init__(self, technicians_df: pd.DataFrame, calendar_df: pd.DataFrame):
        self.technicians = technicians_df
        self.calendar = calendar_df
        self.schedule = {}  # Will hold the optimized schedule
        self.scheduled_times = {}  # Track what's been scheduled: {tech_id: {date: [(start, end)]}}
        self._build_availability_map()
    
    def _build_availability_map(self):
        """Build a map of available time slots per technician per day."""
        print("\nBuilding availability map...")
        
        # Normalize dates to just date (no time component)
        self.calendar['date_normalized'] = pd.to_datetime(self.calendar['date']).dt.date
        
        # Group calendar by technician and date
        self.availability = {}
        
        for tech_id in self.technicians['technician_id'].unique():
            self.availability[tech_id] = {}
            tech_calendar = self.calendar[self.calendar['technician_id'] == tech_id]
            
            for date in tech_calendar['date_normalized'].unique():
                day_events = tech_calendar[tech_calendar['date_normalized'] == date].copy()
                day_events = day_events.sort_values('start_time')
                
                # Store available blocks
                available_blocks = day_events[day_events['event_type'] == 'AVAILABLE'].copy()
                self.availability[tech_id][date] = available_blocks
        
        print(f"✓ Built availability for {len(self.availability)} technicians")
    
    def can_schedule(self, tech_id: str, date, start_minutes: int, 
                    duration: int) -> bool:
        """Check if a technician can be scheduled at given time."""
        # Normalize date
        if hasattr(date, 'date'):
            date = date.date()
        
        end_minutes = start_minutes + duration
        
        if tech_id not in self.availability or date not in self.availability[tech_id]:
            return False
        
        available_blocks = self.availability[tech_id][date]
        
        for _, block in available_blocks.iterrows():
            block_start = time_to_minutes(block['start_time'])
            block_end = time_to_minutes(block['end_time'])
            
            if start_minutes >= block_start and end_minutes <= block_end:
                return True
        
        return False
    
    def find_best_time_slot(self, tech_id: str, date, duration: int,
                           preferred_window: str = None) -> Tuple[int, float]:
        """
        Find best available time slot for a job that doesn't overlap with already scheduled jobs.
        Returns (start_minutes, availability_score).
        """
        # Normalize date
        if hasattr(date, 'date'):
            date = date.date()
            
        if tech_id not in self.availability or date not in self.availability[tech_id]:
            return None, 0.0
        
        # Get already scheduled times for this tech on this day
        scheduled = []
        if tech_id in self.scheduled_times and date in self.scheduled_times[tech_id]:
            scheduled = self.scheduled_times[tech_id][date]
        
        available_blocks = self.availability[tech_id][date]
        best_start = None
        best_score = 0.0
        
        for _, block in available_blocks.iterrows():
            block_start = time_to_minutes(block['start_time'])
            block_end = time_to_minutes(block['end_time'])
            
            # Try different start times within this block
            current_time = block_start
            while current_time + duration <= block_end:
                # Check if this time slot overlaps with any scheduled job
                overlaps = False
                for (sched_start, sched_end) in scheduled:
                    if not (current_time + duration <= sched_start or current_time >= sched_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    # This slot is available!
                    score = 1.0
                    
                    # Prefer slots that match customer preference
                    if preferred_window and not pd.isna(preferred_window):
                        if 'morning' in str(preferred_window).lower() and current_time < 720:
                            score += 0.3
                        elif 'afternoon' in str(preferred_window).lower() and current_time >= 720:
                            score += 0.3
                    
                    # Prefer earlier slots
                    score += 0.1 * (1 - (current_time / 1080))
                    
                    if score > best_score:
                        best_score = score
                        best_start = current_time
                        # Don't break - keep looking for better slots
                
                # Move to next potential start time (15 min increments)
                current_time += 15
        
        return best_start, best_score
    
    def mark_time_scheduled(self, tech_id: str, date, start_minutes: int, end_minutes: int):
        """Mark a time slot as scheduled."""
        if hasattr(date, 'date'):
            date = date.date()
        
        if tech_id not in self.scheduled_times:
            self.scheduled_times[tech_id] = {}
        if date not in self.scheduled_times[tech_id]:
            self.scheduled_times[tech_id][date] = []
        
        self.scheduled_times[tech_id][date].append((start_minutes, end_minutes))

# ============================================================================
# OPTIMIZER
# ============================================================================

class TechnicianDispatchOptimizer:
    """Main optimizer that assigns technicians to work orders."""
    
    def __init__(self, data_loader: DataLoader):
        self.data = data_loader
        self.skill_extractor = SkillExtractor()
        self.distance_calc = DistanceCalculator(data_loader.locations)
        self.availability_mgr = AvailabilityManager(
            data_loader.technicians, 
            data_loader.calendar
        )
        self.assignments = []
        self.jobs_per_tech_per_day = {}  # Track: {tech_id: {date: job_count}}
    
    def can_assign_job_to_tech(self, tech_id: str, date, allow_overflow: bool = False) -> bool:
        """Check if technician can take another job on this day (respects max_jobs_per_day).
        
        Args:
            tech_id: Technician identifier
            date: Date to check
            allow_overflow: If True and CONSTRAINT_MODE='flexible', allow +1 over limit
        """
        # Normalize date
        if hasattr(date, 'date'):
            date = date.date()
        
        # Get technician's max jobs per day
        tech_row = self.data.technicians[
            self.data.technicians['technician_id'] == tech_id
        ].iloc[0]
        max_jobs = int(tech_row['max_jobs_per_day'])
        
        # In flexible mode during overflow reassignment, allow +1 over limit
        if allow_overflow and CONSTRAINT_MODE == 'flexible':
            max_jobs += 1
        
        # Count current jobs for this tech on this day
        if tech_id not in self.jobs_per_tech_per_day:
            return True
        if date not in self.jobs_per_tech_per_day[tech_id]:
            return True
        
        current_count = self.jobs_per_tech_per_day[tech_id][date]
        return current_count < max_jobs
    
    def increment_job_count(self, tech_id: str, date):
        """Increment job count for technician on given date."""
        if hasattr(date, 'date'):
            date = date.date()
        
        if tech_id not in self.jobs_per_tech_per_day:
            self.jobs_per_tech_per_day[tech_id] = {}
        if date not in self.jobs_per_tech_per_day[tech_id]:
            self.jobs_per_tech_per_day[tech_id][date] = 0
        
        self.jobs_per_tech_per_day[tech_id][date] += 1
    
    def calculate_composite_score(self, workorder: pd.Series, tech_id: str,
                                  date, start_time: int, 
                                  prev_job_location: str = None) -> Tuple[float, Dict]:
        """
        Calculate composite score for assigning a work order to a technician.
        Returns (total_score, score_breakdown).
        """
        # Normalize date
        if hasattr(date, 'date'):
            date = date.date()
            
        scores = {}
        
        # 1. SKILL SCORE
        # Use ground truth skills if available, otherwise fall back to NLP extraction
        if 'required_skills_ground_truth' in workorder and not pd.isna(workorder['required_skills_ground_truth']):
            required_skills = self.skill_extractor.parse_ground_truth_skills(workorder['required_skills_ground_truth'])
        else:
            required_skills = self.skill_extractor.extract_skills(workorder['job_description'])
        
        tech_row = self.data.technicians[
            self.data.technicians['technician_id'] == tech_id
        ].iloc[0]
        
        skill_score = self.skill_extractor.calculate_skill_match_score(
            required_skills, 
            tech_row['skills']
        )
        
        # Bonus for previous service history
        history = self.data.service_history[
            (self.data.service_history['customer_id'] == workorder['customer_id']) &
            (self.data.service_history['previous_technician_id'] == tech_id)
        ]
        if not history.empty and history.iloc[0]['satisfaction_score_1to5'] >= 4:
            skill_score = min(1.0, skill_score + 0.2)
        
        scores['skill'] = skill_score
        
        # 2. AVAILABILITY SCORE
        duration = int(workorder['job_duration_minutes'])
        _, avail_score = self.availability_mgr.find_best_time_slot(
            tech_id, date, duration, workorder.get('customer_preferred_window')
        )
        scores['availability'] = avail_score
        
        # 3. TRAVEL SCORE
        travel_score = 0.5  # Default neutral score
        if prev_job_location:
            distance = self.distance_calc.get_distance(
                prev_job_location, 
                workorder['workorder_id']
            )
            # Score inversely proportional to distance (max 50km assumed)
            travel_score = max(0.0, 1.0 - (distance / 50.0))
        
        scores['travel'] = travel_score
        
        # Calculate weighted composite score
        total_score = (
            SKILL_WEIGHT * scores['skill'] +
            AVAILABILITY_WEIGHT * scores['availability'] +
            TRAVEL_WEIGHT * scores['travel']
        )
        
        return total_score, scores
    
    def optimize_schedule_by_day_and_route(self):
        """Optimize schedule with proper route sequencing per technician per day."""
        print("\n" + "="*80)
        print("STARTING TWO-PHASE OPTIMIZATION")
        print("="*80)
        
        workorders = self.data.workorders.copy()
        
        # PHASE 1: Assign technicians based on skills and availability
        print("\nPhase 1: Initial technician assignment...")
        tech_assignments = {}  # {date: {tech_id: [workorders]}}
        
        for idx, wo in workorders.iterrows():
            orig_tech = wo.get('original_assigned_technician_id', None)
            orig_date = wo['scheduled_date']
            orig_start = wo.get('scheduled_start_time', None)
            orig_end = wo.get('scheduled_end_time', None)
            
            # Find best technician based on skills
            best_tech = None
            best_skill_score = -1
            
            for _, tech in self.data.technicians.iterrows():
                tech_id = tech['technician_id']
                
                # Check if technician has any availability this day
                start_time, avail_score = self.availability_mgr.find_best_time_slot(
                    tech_id, orig_date, int(wo['job_duration_minutes']),
                    wo.get('customer_preferred_window')
                )
                
                # Check if technician has capacity for another job this day
                if start_time is not None and avail_score > 0 and self.can_assign_job_to_tech(tech_id, orig_date):
                    # Calculate skill score only
                    # Use ground truth skills if available
                    if 'required_skills_ground_truth' in wo and not pd.isna(wo['required_skills_ground_truth']):
                        required_skills = self.skill_extractor.parse_ground_truth_skills(wo['required_skills_ground_truth'])
                    else:
                        required_skills = self.skill_extractor.extract_skills(wo['job_description'])
                    
                    skill_score = self.skill_extractor.calculate_skill_match_score(
                        required_skills, tech['skills']
                    )
                    
                    # Bonus for previous service
                    history = self.data.service_history[
                        (self.data.service_history['customer_id'] == wo['customer_id']) &
                        (self.data.service_history['previous_technician_id'] == tech_id)
                    ]
                    if not history.empty and history.iloc[0]['satisfaction_score_1to5'] >= 4:
                        skill_score = min(1.0, skill_score + 0.2)
                    
                    if skill_score > best_skill_score:
                        best_skill_score = skill_score
                        best_tech = tech_id
            
            if best_tech:
                if orig_date not in tech_assignments:
                    tech_assignments[orig_date] = {}
                if best_tech not in tech_assignments[orig_date]:
                    tech_assignments[orig_date][best_tech] = []
                
                tech_assignments[orig_date][best_tech].append({
                    'workorder': wo,
                    'orig_tech': orig_tech,
                    'orig_start': orig_start,
                    'orig_end': orig_end,
                    'skill_score': best_skill_score
                })
                
                # In strict mode only: Reserve capacity for this tentative assignment
                if CONSTRAINT_MODE == 'strict':
                    self.increment_job_count(best_tech, orig_date)
        
        # PHASE 2: Route optimization - sequence jobs by distance for each tech/day
        print("\nPhase 2: Route optimization and time sequencing...")
        
        unassigned_jobs = []  # Track jobs that couldn't fit
        
        for date in sorted(tech_assignments.keys()):
            for tech_id in tech_assignments[date]:
                jobs = tech_assignments[date][tech_id]
                if len(jobs) == 0:
                    continue
                
                print(f"\n  Optimizing route for {tech_id} on {date}: {len(jobs)} jobs")
                
                # Use nearest neighbor algorithm to sequence jobs
                sequenced_jobs = self._sequence_jobs_by_distance(jobs)
                
                # Assign time slots to sequenced jobs
                current_time = 480  # Start at 8:00 AM (08:00)
                shift_end = 1080     # End at 6:00 PM (18:00)
                
                for job_info in sequenced_jobs:
                    wo = job_info['workorder']
                    duration = int(wo['job_duration_minutes'])
                    
                    # Find next available slot starting from current_time
                    start_time, avail_score = self.availability_mgr.find_best_time_slot(
                        tech_id, date, duration, wo.get('customer_preferred_window')
                    )
                    
                    if start_time is None:
                        start_time = current_time
                    else:
                        # Use the found slot if it's after current time, otherwise use current
                        start_time = max(start_time, current_time)
                    
                    end_time = start_time + duration
                    
                    # CRITICAL: Enforce shift hours (08:00-18:00)
                    if end_time > shift_end:
                        # Job would end after shift - try to find another technician
                        print(f"    ⚠ {wo['workorder_id']} would end at {minutes_to_time(end_time)} (after 18:00)")
                        # In strict mode, release the reserved capacity since job didn't schedule
                        if CONSTRAINT_MODE == 'strict':
                            if tech_id in self.jobs_per_tech_per_day and date in self.jobs_per_tech_per_day[tech_id]:
                                self.jobs_per_tech_per_day[tech_id][date] -= 1
                        unassigned_jobs.append(job_info)
                        continue
                    
                    if start_time < 480:  # Before 08:00
                        start_time = 480
                        end_time = start_time + duration
                        if end_time > shift_end:
                            print(f"    ⚠ {wo['workorder_id']} too long to fit in shift")
                            # In strict mode, release the reserved capacity since job didn't schedule
                            if CONSTRAINT_MODE == 'strict':
                                if tech_id in self.jobs_per_tech_per_day and date in self.jobs_per_tech_per_day[tech_id]:
                                    self.jobs_per_tech_per_day[tech_id][date] -= 1
                            unassigned_jobs.append(job_info)
                            continue
                    
                    # Calculate final scores
                    travel_score = job_info.get('travel_score', 0.5)
                    total_score = (
                        SKILL_WEIGHT * job_info['skill_score'] +
                        AVAILABILITY_WEIGHT * min(1.0, avail_score) +
                        TRAVEL_WEIGHT * travel_score
                    )
                    
                    rationale = f"Skill:{job_info['skill_score']:.2f} Avail:{min(1.0, avail_score):.2f} Travel:{travel_score:.2f} | Total:{total_score:.2f}"
                    
                    assignment = {
                        'workorder_id': wo['workorder_id'],
                        'original_assigned_technician_id': job_info['orig_tech'],
                        'original_scheduled_date': date,
                        'original_start_time': job_info['orig_start'],
                        'original_end_time': job_info['orig_end'],
                        'optimized_assigned_technician_id': tech_id,
                        'optimized_scheduled_date': date,
                        'optimized_start_time': minutes_to_time(start_time),
                        'optimized_end_time': minutes_to_time(end_time),
                        'rationale': rationale
                    }
                    
                    self.assignments.append(assignment)
                    
                    # Mark this time as scheduled to prevent overlaps
                    self.availability_mgr.mark_time_scheduled(tech_id, date, start_time, end_time)
                    
                    # Increment job count (in strict mode, this was already done in Phase 1)
                    if CONSTRAINT_MODE == 'flexible':
                        self.increment_job_count(tech_id, date)
                    
                    # Reduce buffer - use actual end time to maximize capacity
                    current_time = end_time
        
        # PHASE 3: Reassign jobs that didn't fit
        if unassigned_jobs:
            print(f"\nPhase 3: Reassigning {len(unassigned_jobs)} jobs that didn't fit...")
            self._reassign_overflow_jobs(unassigned_jobs, tech_assignments)
        
        print(f"\n✓ Optimization complete: {len(self.assignments)} assignments made")
        return self
    
    def _sequence_jobs_by_distance(self, jobs):
        """Sequence jobs using nearest neighbor algorithm to minimize travel."""
        if len(jobs) <= 1:
            return jobs
        
        sequenced = []
        remaining = jobs.copy()
        
        # Start with first job
        current_job = remaining.pop(0)
        sequenced.append(current_job)
        current_location = current_job['workorder']['workorder_id']
        
        # Greedily pick nearest job
        while remaining:
            nearest_job = None
            nearest_distance = float('inf')
            nearest_idx = -1
            
            for idx, job in enumerate(remaining):
                next_location = job['workorder']['workorder_id']
                distance = self.distance_calc.get_distance(current_location, next_location)
                
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_job = job
                    nearest_idx = idx
            
            if nearest_job:
                # Calculate travel score for this job
                travel_score = max(0.0, 1.0 - (nearest_distance / 50.0))
                nearest_job['travel_score'] = travel_score
                nearest_job['travel_distance'] = nearest_distance
                
                sequenced.append(nearest_job)
                current_location = nearest_job['workorder']['workorder_id']
                remaining.pop(nearest_idx)
            else:
                break
        
        return sequenced
    
    def _reassign_overflow_jobs(self, unassigned_jobs, tech_assignments):
        """Try to reassign jobs that didn't fit in their original assignment."""
        shift_end = 1080  # 18:00
        
        print(f"\n  Trying to reassign {len(unassigned_jobs)} overflow jobs...")
        
        # Sort by job duration (shortest first for better packing)
        unassigned_jobs = sorted(unassigned_jobs, key=lambda x: int(x['workorder']['job_duration_minutes']))
        
        # Get all dates in the schedule
        all_dates = sorted(set([pd.to_datetime(d).date() for d in pd.read_excel(f"{self.data.data_dir}/04_workorders_week_original.xlsx")['scheduled_date'].unique()]))
        
        for job_info in unassigned_jobs:
            wo = job_info['workorder']
            duration = int(wo['job_duration_minutes'])
            orig_date = job_info['workorder']['scheduled_date']
            
            assigned = False
            
            # Strategy 1: Try ALL technicians on the same day with ANY available slot
            for _, tech in self.data.technicians.iterrows():
                tech_id = tech['technician_id']
                
                # Check if technician has capacity
                if not self.can_assign_job_to_tech(tech_id, orig_date):
                    continue
                
                start_time, avail_score = self.availability_mgr.find_best_time_slot(
                    tech_id, orig_date, duration, None  # Ignore customer preference for overflow
                )
                
                if start_time and (start_time + duration) <= shift_end:
                    assigned = self._make_assignment(job_info, tech_id, orig_date, start_time, duration, avail_score, "Same day")
                    break
            
            # Strategy 2: Try ALL days in the week, ALL technicians
            if not assigned:
                for target_date in all_dates:
                    for _, tech in self.data.technicians.iterrows():
                        tech_id = tech['technician_id']
                        
                        # Check if technician has capacity
                        if not self.can_assign_job_to_tech(tech_id, target_date):
                            continue
                        
                        start_time, avail_score = self.availability_mgr.find_best_time_slot(
                            tech_id, target_date, duration, None
                        )
                        
                        if start_time and (start_time + duration) <= shift_end:
                            days_diff = abs((pd.to_datetime(target_date) - pd.to_datetime(orig_date)).days)
                            assigned = self._make_assignment(job_info, tech_id, target_date, start_time, duration, avail_score, f"Moved to {target_date}")
                            break
                    
                    if assigned:
                        break
            
            # Strategy 3: Force fit into any gap (last resort) - allow +1 overflow in flexible mode
            if not assigned:
                for target_date in all_dates:
                    for _, tech in self.data.technicians.iterrows():
                        tech_id = tech['technician_id']
                        
                        # Check if technician has capacity (allow overflow in flexible mode)
                        if not self.can_assign_job_to_tech(tech_id, target_date, allow_overflow=True):
                            continue
                        
                        # Try every 15-minute slot in the shift
                        for start_time in range(480, shift_end - duration + 1, 15):
                            # Check if this exact slot is free
                            if hasattr(orig_date, 'date'):
                                check_date = target_date
                            else:
                                check_date = target_date
                            
                            # Get scheduled times
                            scheduled = []
                            if tech_id in self.availability_mgr.scheduled_times and check_date in self.availability_mgr.scheduled_times[tech_id]:
                                scheduled = self.availability_mgr.scheduled_times[tech_id][check_date]
                            
                            # Check overlap
                            end_time = start_time + duration
                            is_free = True
                            for (sched_start, sched_end) in scheduled:
                                if not (end_time <= sched_start or start_time >= sched_end):
                                    is_free = False
                                    break
                            
                            if is_free and end_time <= shift_end:
                                assigned = self._make_assignment(job_info, tech_id, check_date, start_time, duration, 0.5, f"Force-fit {check_date}")
                                break
                        
                        if assigned:
                            break
                    
                    if assigned:
                        break
            
            if not assigned:
                print(f"    ✗ Could not reassign {wo['workorder_id']} - truly no capacity")
    
    def _make_assignment(self, job_info, tech_id, date, start_time, duration, avail_score, reason):
        """Helper to create and record an assignment."""
        wo = job_info['workorder']
        end_time = start_time + duration
        
        # Calculate skill score
        # Use ground truth skills if available
        if 'required_skills_ground_truth' in wo and not pd.isna(wo['required_skills_ground_truth']):
            required_skills = self.skill_extractor.parse_ground_truth_skills(wo['required_skills_ground_truth'])
        else:
            required_skills = self.skill_extractor.extract_skills(wo['job_description'])
        
        tech_row = self.data.technicians[self.data.technicians['technician_id'] == tech_id].iloc[0]
        skill_score = self.skill_extractor.calculate_skill_match_score(
            required_skills, tech_row['skills']
        )
        
        travel_score = 0.5  # Neutral for reassigned
        total_score = (
            SKILL_WEIGHT * skill_score +
            AVAILABILITY_WEIGHT * min(1.0, avail_score) +
            TRAVEL_WEIGHT * travel_score
        )
        
        rationale = f"Skill:{skill_score:.2f} Avail:{min(1.0, avail_score):.2f} Travel:{travel_score:.2f} | Total:{total_score:.2f} ({reason})"
        
        assignment = {
            'workorder_id': wo['workorder_id'],
            'original_assigned_technician_id': job_info['orig_tech'],
            'original_scheduled_date': job_info['workorder']['scheduled_date'],
            'original_start_time': job_info['orig_start'],
            'original_end_time': job_info['orig_end'],
            'optimized_assigned_technician_id': tech_id,
            'optimized_scheduled_date': date,
            'optimized_start_time': minutes_to_time(start_time),
            'optimized_end_time': minutes_to_time(end_time),
            'rationale': rationale
        }
        
        self.assignments.append(assignment)
        self.availability_mgr.mark_time_scheduled(tech_id, date, start_time, end_time)
        self.increment_job_count(tech_id, date)
        print(f"    ✓ {wo['workorder_id']} → {tech_id} on {date} ({reason})")
        return True
    
    def optimize_schedule(self):
        """Main optimization entry point."""
        return self.optimize_schedule_by_day_and_route()
    
    def generate_output(self, output_file: str):
        """Generate final_schedule.csv output."""
        print(f"\nGenerating output file: {output_file}")
        
        df = pd.DataFrame(self.assignments)
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved {len(df)} assignments to {output_file}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total work orders processed: {len(df)}")
        
        if len(df) > 0:
            print(f"Technician changes: {(df['original_assigned_technician_id'] != df['optimized_assigned_technician_id']).sum()}")
            print(f"Time changes: {(df['original_start_time'] != df['optimized_start_time']).sum()}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("SMART TECHNICIAN DISPATCH OPTIMIZER")
    print("TELUS CAIO B2B Hackathon #1")
    print("="*80)
    
    # Data directory
    data_dir = "data"
    
    # Load data
    loader = DataLoader(data_dir)
    loader.load_all()
    
    # Run optimization
    optimizer = TechnicianDispatchOptimizer(loader)
    optimizer.optimize_schedule()
    
    # Generate output
    output_file = "output/final_schedule.csv"
    optimizer.generate_output(output_file)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
