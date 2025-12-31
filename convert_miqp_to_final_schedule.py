"""
Convert MIQP optimization results to final_schedule.csv format.

Reference format from existing final_schedule.csv:
- Columns: workorder_id, original_assigned_technician_id, original_scheduled_date, 
  original_start_time, original_end_time, optimized_assigned_technician_id, 
  optimized_scheduled_date, optimized_start_time, optimized_end_time, rationale
- Rationale format: "Skill:0.67 Avail:1.00 Travel:0.50 | Total:0.71"
"""

import pandas as pd
import numpy as np


def calculate_availability_score(start_slot, pref_start, pref_end, N=40):
    """Calculate availability score based on customer preferred time window."""
    if pref_start <= start_slot <= pref_end:
        return 1.0
    else:
        deviation = pref_start - start_slot if start_slot < pref_start else start_slot - pref_end
        return max(0.0, 1.0 - (deviation / N))


def calculate_travel_score(worker_dist, max_distance):
    """Calculate travel score based on actual distance traveled."""
    if max_distance == 0:
        return 1.0
    return max(0.0, 1.0 - (worker_dist / max_distance))


def convert_assignments_to_final_schedule(assignments, orders_df, workers_df, skill_match, 
                                         preferred_windows, worker_distances, scheduled_date, N=40):
    """
    Convert MIQP assignment results to final_schedule.csv format.
    
    Args:
        assignments: list of dicts with assignment details
        orders_df: DataFrame of work orders for the day
        workers_df: DataFrame of technicians
        skill_match: numpy array (M x K) of skill match values [0,1]
        preferred_windows: list of tuples (pref_start, pref_end) for each order
        worker_distances: dict mapping technician_id to total distance traveled (km)
        scheduled_date: date string in 'YYYY-MM-DD' format
        N: total number of time slots
    
    Returns:
        DataFrame in final_schedule.csv format
    """
    tech_id_to_idx = {workers_df.iloc[m]['technician_id']: m for m in range(len(workers_df))}
    max_distance = max(worker_distances.values()) if worker_distances else 1.0
    
    result_rows = []
    
    for assignment in assignments:
        order_idx = assignment['order_index']
        workorder_id = assignment['workorder_id']
        order_row = orders_df.iloc[order_idx]
        
        # Original assignment info
        original_tech_id = order_row['original_assigned_technician_id']
        original_date = order_row['scheduled_date']
        
        # Optimized assignment info
        optimized_tech_id = assignment['technician_id']
        optimized_date = f"{scheduled_date} 00:00:00"
        optimized_start = assignment['start_time']
        optimized_end = assignment['end_time']
        
        # Calculate rationale scores
        tech_idx = tech_id_to_idx.get(optimized_tech_id)
        skill_score = skill_match[tech_idx, order_idx] if tech_idx is not None else 0.0
        
        pref_start, pref_end = preferred_windows[order_idx]
        start_slot = assignment['start_slot']
        avail_score = calculate_availability_score(start_slot, pref_start, pref_end, N)
        
        worker_dist = worker_distances.get(optimized_tech_id, 0.0)
        travel_score = calculate_travel_score(worker_dist, max_distance)
        
        # Total score: weighted average (35% skill, 30% avail, 35% travel)
        total_score = 0.35 * skill_score + 0.30 * avail_score + 0.35 * travel_score
        
        rationale = (f"Skill:{skill_score:.2f} "
                    f"Avail:{avail_score:.2f} "
                    f"Travel:{travel_score:.2f} | "
                    f"Total:{total_score:.2f}")
        
        result_rows.append({
            'workorder_id': workorder_id,
            'original_assigned_technician_id': original_tech_id,
            'original_scheduled_date': original_date,
            'original_start_time': '',
            'original_end_time': '',
            'optimized_assigned_technician_id': optimized_tech_id,
            'optimized_scheduled_date': optimized_date,
            'optimized_start_time': optimized_start,
            'optimized_end_time': optimized_end,
            'rationale': rationale
        })
    
    return pd.DataFrame(result_rows)
