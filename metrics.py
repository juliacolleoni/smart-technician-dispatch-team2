"""
Quality Metrics for Technician Dispatch Optimization
TELUS CAIO B2B Hackathon #1
December 29-31, 2025

This module calculates three key quality metrics:
1. Skill Match Score - How well technician skills match job requirements
2. Schedule Preference Adherence - How well schedules respect customer preferences
3. Travel Efficiency - Average distance traveled between jobs
"""

import pandas as pd
import numpy as np
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from technician_dispatch_optimizer import TechnicianDispatchOptimizer

# Import utility functions
from technician_dispatch_optimizer import time_to_minutes, minutes_to_time, haversine_distance, SKILL_WEIGHT, AVAILABILITY_WEIGHT, TRAVEL_WEIGHT


class QualityMetrics:
    """Calculate quality metrics for assignment evaluation."""
    
    def __init__(self, optimizer: 'TechnicianDispatchOptimizer'):
        self.optimizer = optimizer
        self.skill_extractor = optimizer.skill_extractor
        self.distance_calc = optimizer.distance_calc
    
    def calculate_skill_match_score(self):
        """Métrica 1: Average skill match quality (0-1)."""
        scores = []
        
        for assignment in self.optimizer.assignments:
            wo_id = assignment['workorder_id']
            tech_id = assignment['optimized_assigned_technician_id']
            
            # Get work order
            wo = self.optimizer.data.workorders[
                self.optimizer.data.workorders['workorder_id'] == wo_id
            ]
            
            if wo.empty:
                continue
            
            wo = wo.iloc[0]
            
            # Extract required skills
            if 'required_skills_ground_truth' in wo and not pd.isna(wo['required_skills_ground_truth']):
                required_skills = self.skill_extractor.parse_ground_truth_skills(wo['required_skills_ground_truth'])
            else:
                required_skills = self.skill_extractor.extract_skills(wo['job_description'])
            
            # Get technician skills
            tech = self.optimizer.data.technicians[
                self.optimizer.data.technicians['technician_id'] == tech_id
            ].iloc[0]
            
            # Calculate match
            skill_score = self.skill_extractor.calculate_skill_match_score(
                required_skills, 
                tech['skills']
            )
            
            scores.append(skill_score)
        
        return {
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'min': np.min(scores) if scores else 0,
            'max': np.max(scores) if scores else 0,
            'scores': scores
        }
    
    def calculate_schedule_preference_score(self):
        """Métrica 2: Average schedule preference adherence (0-1)."""
        scores = []
        
        for assignment in self.optimizer.assignments:
            wo_id = assignment['workorder_id']
            scheduled_start = assignment['optimized_start_time']
            
            # Get work order
            wo = self.optimizer.data.workorders[
                self.optimizer.data.workorders['workorder_id'] == wo_id
            ]
            
            if wo.empty:
                continue
            
            wo = wo.iloc[0]
            customer_pref = wo.get('customer_preferred_window')
            
            # Check if UNSCHEDULED (original assignments that couldn't be scheduled)
            if scheduled_start == 'UNSCHEDULED':
                score = 0.0  # Failed to schedule in original system
            # Calculate overlap score
            elif pd.isna(customer_pref) or customer_pref == '':
                score = 0.5  # Neutral if no preference
            else:
                pref_str = str(customer_pref).strip()
                scheduled_minutes = time_to_minutes(scheduled_start)
                
                # Parse specific time window (e.g., "08:00-12:00")
                if '-' in pref_str and ':' in pref_str:
                    try:
                        parts = pref_str.split('-')
                        pref_start = time_to_minutes(parts[0].strip())
                        pref_end = time_to_minutes(parts[1].strip())
                        
                        # Check if scheduled time is within preferred window
                        if scheduled_minutes >= pref_start and scheduled_minutes <= pref_end:
                            score = 1.0  # Perfect match
                        elif scheduled_minutes < pref_start:
                            # Before preferred window
                            gap = pref_start - scheduled_minutes
                            score = max(0, 1.0 - (gap / 120))  # Penalize up to 2h gap
                        else:
                            # After preferred window
                            gap = scheduled_minutes - pref_end
                            score = max(0, 1.0 - (gap / 120))
                    except:
                        score = 0.5
                else:
                    # Generic morning/afternoon preference
                    pref_lower = pref_str.lower()
                    if 'morning' in pref_lower and scheduled_minutes < 720:
                        score = 1.0
                    elif 'afternoon' in pref_lower and scheduled_minutes >= 720:
                        score = 1.0
                    else:
                        score = 0.3  # Missed preference
            
            scores.append(score)
        
        return {
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'min': np.min(scores) if scores else 0,
            'max': np.max(scores) if scores else 0,
            'scores': scores
        }
    
    def calculate_travel_efficiency_score(self):
        """Métrica 3: Average travel efficiency (distance normalized to 0-1)."""
        distances = []
        
        # Group assignments by technician and date
        tech_assignments = {}
        for assignment in self.optimizer.assignments:
            tech_id = assignment['optimized_assigned_technician_id']
            date = assignment['optimized_scheduled_date']
            
            if tech_id not in tech_assignments:
                tech_assignments[tech_id] = {}
            if date not in tech_assignments[tech_id]:
                tech_assignments[tech_id][date] = []
            
            tech_assignments[tech_id][date].append(assignment)
        
        # Calculate distances
        for tech_id, dates in tech_assignments.items():
            # Get technician home location
            tech = self.optimizer.data.technicians[
                self.optimizer.data.technicians['technician_id'] == tech_id
            ].iloc[0]
            
            for date, assignments in dates.items():
                # Sort by start time
                assignments_sorted = sorted(
                    assignments, 
                    key=lambda x: time_to_minutes(x['optimized_start_time'])
                )
                
                if not assignments_sorted:
                    continue
                
                # Home to first job
                first_job_id = assignments_sorted[0]['workorder_id']
                first_job_loc = self.optimizer.data.locations[
                    self.optimizer.data.locations['node_id'] == first_job_id
                ]
                
                if not first_job_loc.empty:
                    first_job_loc = first_job_loc.iloc[0]
                    dist = haversine_distance(
                        tech['home_base_lat'], tech['home_base_lon'],
                        first_job_loc['lat'], first_job_loc['lon']
                    )
                    distances.append(dist)
                
                # Job to job
                for i in range(len(assignments_sorted) - 1):
                    job1_id = assignments_sorted[i]['workorder_id']
                    job2_id = assignments_sorted[i+1]['workorder_id']
                    
                    dist = self.distance_calc.get_distance(job1_id, job2_id)
                    distances.append(dist)
                
                # Last job to home
                last_job_id = assignments_sorted[-1]['workorder_id']
                last_job_loc = self.optimizer.data.locations[
                    self.optimizer.data.locations['node_id'] == last_job_id
                ]
                
                if not last_job_loc.empty:
                    last_job_loc = last_job_loc.iloc[0]
                    dist = haversine_distance(
                        last_job_loc['lat'], last_job_loc['lon'],
                        tech['home_base_lat'], tech['home_base_lon']
                    )
                    distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        
        # Normalize: 50 km = worst case (score 0), 0 km = best case (score 1)
        max_reasonable_distance = 50.0
        efficiency_score = max(0, 1.0 - (avg_distance / max_reasonable_distance))
        
        return {
            'avg_distance_km': avg_distance,
            'efficiency_score': efficiency_score,
            'total_distance_km': sum(distances) if distances else 0,
            'std': np.std(distances) if distances else 0,
            'min': np.min(distances) if distances else 0,
            'max': np.max(distances) if distances else 0,
            'distances': distances
        }
    
    def calculate_overall_quality_score(self, use_dynamic_weights=True, static_weights=None):
        """
        Calculate weighted overall quality score with dynamic or static weights.
        
        Args:
            use_dynamic_weights: Whether to use dynamic weights
            static_weights: Optional custom static weights
        
        Returns:
            dict: All metrics and the combined score
        """
        if static_weights is None:
            static_weights = {
                'skill': SKILL_WEIGHT,
                'schedule': AVAILABILITY_WEIGHT,
                'travel': TRAVEL_WEIGHT
            }
        
        # Calculate individual metrics
        skill = self.calculate_skill_match_score()
        schedule = self.calculate_schedule_preference_score()
        travel = self.calculate_travel_efficiency_score()
        
        if not use_dynamic_weights:
            # Use static weights
            weights_used = static_weights
            
            overall = (
                weights_used['skill'] * skill['mean'] +
                weights_used['schedule'] * schedule['mean'] +
                weights_used['travel'] * travel['efficiency_score']
            )
        else:
            # Use dynamic weights for each work order
            total_weighted_score = 0.0
            total_weight = 0.0
            all_weights = []
            
            for assignment in self.optimizer.assignments:
                wo_id = assignment['workorder_id']
                
                # Get work order
                wo_data = self.optimizer.data.workorders[
                    self.optimizer.data.workorders['workorder_id'] == wo_id
                ]
                
                if wo_data.empty:
                    continue
                    
                wo = wo_data.iloc[0]
                
                # Get dynamic weights for this job
                job_weights = self.get_dynamic_weights_for_job(wo)
                
                # Find job-specific metrics if possible
                job_skill = next((s for i, s in enumerate(skill['scores']) 
                                if i < len(self.optimizer.assignments) and 
                                self.optimizer.assignments[i]['workorder_id'] == wo_id), 
                                skill['mean'])
                
                job_schedule = next((s for i, s in enumerate(schedule['scores']) 
                                    if i < len(self.optimizer.assignments) and 
                                    self.optimizer.assignments[i]['workorder_id'] == wo_id), 
                                    schedule['mean'])
                
                # Calculate job-specific score
                job_score = (
                    job_weights['skill'] * job_skill +
                    job_weights['schedule'] * job_schedule +
                    job_weights['travel'] * travel['efficiency_score']
                )
                
                total_weighted_score += job_score
                total_weight += 1.0
                all_weights.append(job_weights)
            
            # Calculate overall score as average of job scores
            if total_weight > 0:
                overall = total_weighted_score / total_weight
            else:
                overall = 0.0
                
            # Calculate average weights
            weights_used = {
                'skill': sum(w['skill'] for w in all_weights) / len(all_weights) if all_weights else static_weights['skill'],
                'schedule': sum(w['schedule'] for w in all_weights) / len(all_weights) if all_weights else static_weights['schedule'],
                'travel': sum(w['travel'] for w in all_weights) / len(all_weights) if all_weights else static_weights['travel']
            }
        
        # Return all metrics
        return {
            'overall_score': overall,
            'skill_match': skill,
            'schedule_preference': schedule,
            'travel_efficiency': travel,
            'weights_used': weights_used,
            'using_dynamic_weights': use_dynamic_weights
        }

    
    def calculate_metrics_for_original_assignments(self):
        """Calculate metrics using ORIGINAL assignments (before optimization)."""
        # Temporarily swap to use original assignments
        original_assignments = []
        
        # Load calendar data to get original scheduled times
        calendar_df = self.optimizer.data.calendar
        
        for assignment in self.optimizer.assignments:
            wo_id = assignment['workorder_id']
            
            # Get work order to extract customer preference
            wo = self.optimizer.data.workorders[
                self.optimizer.data.workorders['workorder_id'] == wo_id
            ].iloc[0]
            
            # Create a copy with original technician
            original_assignment = assignment.copy()
            original_assignment['optimized_assigned_technician_id'] = assignment['original_assigned_technician_id']
            original_assignment['optimized_scheduled_date'] = assignment['original_scheduled_date']
            
            # Look up original schedule from calendar (table 05)
            orig_start = None
            calendar_entry = calendar_df[
                (calendar_df['workorder_id'] == wo_id) & 
                (calendar_df['event_type'] == 'WORKORDER')
            ]
            
            if not calendar_entry.empty:
                # Found original scheduled time
                orig_start = calendar_entry.iloc[0]['start_time']
                # Convert datetime.time to string if needed
                if hasattr(orig_start, 'hour'):
                    orig_start = f"{orig_start.hour:02d}:{orig_start.minute:02d}"
            else:
                # UNSCHEDULED: Use a marker to indicate it wasn't scheduled
                # This will result in score=0 in schedule preference calculation
                orig_start = 'UNSCHEDULED'
            
            original_assignment['optimized_start_time'] = orig_start
            original_assignments.append(original_assignment)
        
        # Temporarily replace assignments
        temp_assignments = self.optimizer.assignments
        self.optimizer.assignments = original_assignments
        
        # Calculate metrics
        results = self.calculate_overall_quality_score()
        
        # Restore original assignments
        self.optimizer.assignments = temp_assignments
        
        return results
    
    def print_comparison_report(self):
        """Print comparison between original and optimized assignments."""
        print("\n" + "="*80)
        print("QUALITY COMPARISON: ORIGINAL vs OPTIMIZED")
        print("="*80)
        
        # Calculate metrics for both
        original_results = self.calculate_metrics_for_original_assignments()
        optimized_results = self.calculate_overall_quality_score()
        
        # Extract values
        orig_skill = original_results['skill_match']['mean']
        opt_skill = optimized_results['skill_match']['mean']
        
        orig_schedule = original_results['schedule_preference']['mean']
        opt_schedule = optimized_results['schedule_preference']['mean']
        
        orig_travel_score = original_results['travel_efficiency']['efficiency_score']
        opt_travel_score = optimized_results['travel_efficiency']['efficiency_score']
        
        orig_travel_km = original_results['travel_efficiency']['avg_distance_km']
        opt_travel_km = optimized_results['travel_efficiency']['avg_distance_km']
        
        orig_total_km = original_results['travel_efficiency']['total_distance_km']
        opt_total_km = optimized_results['travel_efficiency']['total_distance_km']
        
        orig_overall = original_results['overall_score']
        opt_overall = optimized_results['overall_score']
        
        # Calculate improvements
        skill_improvement = ((opt_skill - orig_skill) / orig_skill * 100) if orig_skill > 0 else 0
        schedule_improvement = ((opt_schedule - orig_schedule) / orig_schedule * 100) if orig_schedule > 0 else 0
        travel_improvement = ((opt_travel_score - orig_travel_score) / orig_travel_score * 100) if orig_travel_score > 0 else 0
        overall_improvement = ((opt_overall - orig_overall) / orig_overall * 100) if orig_overall > 0 else 0
        
        km_saved = orig_total_km - opt_total_km
        km_saved_percent = (km_saved / orig_total_km * 100) if orig_total_km > 0 else 0
        
        # Calculate reassignments
        total_wo = len(self.optimizer.assignments)
        reassigned_count = sum(
            1 for assignment in self.optimizer.assignments
            if assignment['original_assigned_technician_id'] != assignment['optimized_assigned_technician_id']
        )
        reassignment_percent = (reassigned_count / total_wo * 100) if total_wo > 0 else 0
        
        # Print table header
        print("\n                              ORIGINAL    OPTIMIZED    IMPROVEMENT")
        print("-" * 80)
        
        # Skill Match
        arrow = "✓" if opt_skill >= orig_skill else "✗"
        print(f"Skill Match                   {orig_skill:6.1%}      {opt_skill:6.1%}      {skill_improvement:+6.1f}%  {arrow}")
        
        # Schedule Preference
        arrow = "✓" if opt_schedule >= orig_schedule else "✗"
        print(f"Schedule Preference           {orig_schedule:6.1%}      {opt_schedule:6.1%}      {schedule_improvement:+6.1f}%  {arrow}")
        
        # Travel Efficiency Score
        arrow = "✓" if opt_travel_score >= orig_travel_score else "✗"
        print(f"Travel Efficiency (score)     {orig_travel_score:6.1%}      {opt_travel_score:6.1%}      {travel_improvement:+6.1f}%  {arrow}")
        
        # Travel Distance
        arrow = "✓" if opt_travel_km <= orig_travel_km else "✗"
        distance_change = opt_travel_km - orig_travel_km
        print(f"Avg Distance (km/transition)  {orig_travel_km:6.2f}      {opt_travel_km:6.2f}      {distance_change:+6.2f}   {arrow}")
        
        print("-" * 80)
        
        # Overall Score
        arrow = "✓" if opt_overall >= orig_overall else "✗"
        print(f"OVERALL QUALITY SCORE         {orig_overall:6.1%}      {opt_overall:6.1%}      {overall_improvement:+6.1f}%  {arrow}")
        
        print("=" * 80)
        
        # Additional details
        print("\nTRAVEL DISTANCE SUMMARY:")
        arrow = "✓" if km_saved >= 0 else "✗"
        print(f"  Original Total:    {orig_total_km:8.2f} km")
        print(f"  Optimized Total:   {opt_total_km:8.2f} km")
        print(f"  Distance Saved:    {km_saved:8.2f} km ({km_saved_percent:+.1f}%)  {arrow}")
        
        print("\nREASSIGNMENT SUMMARY:")
        print(f"  Total Work Orders: {total_wo:8d}")
        print(f"  Reassigned:        {reassigned_count:8d} ({reassignment_percent:.1f}%)")
        print(f"  Unchanged:         {total_wo - reassigned_count:8d} ({100 - reassignment_percent:.1f}%)")
        
        print("\n" + "="*80)
    
    def print_quality_report(self, use_dynamic_weights=True):
        """Print comprehensive quality metrics report."""
        print("\n" + "="*80)
        print("QUALITY METRICS REPORT")
        print("="*80)
        
        results = self.calculate_overall_quality_score(use_dynamic_weights=use_dynamic_weights)
        
        skill = results['skill_match']
        schedule = results['schedule_preference']
        travel = results['travel_efficiency']
        overall = results['overall_score']
        weights = results['weights_used']
        
        # [Existing print statements for metrics]
        
        print("\n" + "-"*80)
        print(f"OVERALL QUALITY SCORE: {overall:.1%}")
        
        # Show weight information
        if results['using_dynamic_weights']:
            print("\nUSING DYNAMIC CONTEXT-BASED WEIGHTS (average across all jobs):")
        else:
            print("\nUSING STATIC WEIGHTS:")
            
        print(f"   Skill Weight: {weights['skill']:.0%}")
        print(f"   Schedule Weight: {weights['schedule']:.0%}")
        print(f"   Travel Weight: {weights['travel']:.0%}")
        
        print("="*80)


    def get_dynamic_weights_for_job(self, workorder):
        """
        Get dynamic weights for a specific job using the existing DynamicWeightManager.
        
        Args:
            workorder: Work order data from optimizer.data.workorders
            
        Returns:
            Dict with dynamic weights
        """
        # Access the optimizer's DynamicWeightManager
        # If it's a class variable:
        if hasattr(self.optimizer, 'weight_manager'):
            weight_manager = self.optimizer.weight_manager
        else:
            # Create a new instance if needed
            from technician_dispatch_optimizer import DynamicWeightManager
            weight_manager = DynamicWeightManager()
        
        # Extract context for this work order
        context = {}
        
        # Job type from description
        description = str(workorder['job_description']).lower()
        if 'install' in description:
            context['job_type'] = 'installation'
        elif 'repair' in description:
            context['job_type'] = 'repair'
        elif 'upgrade' in description:
            context['job_type'] = 'upgrade'
        else:
            context['job_type'] = 'standard'
            
        # Customer tier if available
        customer_id = workorder.get('customer_id')
        if customer_id is not None:
            customer_data = self.optimizer.data.customers[
                self.optimizer.data.customers['customer_id'] == customer_id
            ]
            if not customer_data.empty:
                context['customer_tier'] = customer_data.iloc[0].get('account_tier', 'standard')
        
        # Time sensitivity from preferred window
        preferred_window = workorder.get('customer_preferred_window')
        if pd.notna(preferred_window) and '-' in str(preferred_window) and ':' in str(preferred_window):
            context['time_sensitivity'] = 'high'
        elif pd.notna(preferred_window):
            context['time_sensitivity'] = 'medium'
        else:
            context['time_sensitivity'] = 'low'
            
        # Get dynamic weights for this context
        dynamic_weights = weight_manager.calculate_weights(context)
        
        # Return weights
        return {
            'skill': dynamic_weights['skill'],
            'schedule': dynamic_weights.get('availability', dynamic_weights.get('preference', AVAILABILITY_WEIGHT)),
            'travel': dynamic_weights['travel']
        }

    def compare_dynamic_vs_static_weights(self):
        """Compare results with dynamic vs. static weights."""
        print("\n" + "="*80)
        print("COMPARISON: DYNAMIC WEIGHTS vs. STATIC WEIGHTS")
        print("="*80)
        
        # Calculate with static weights
        static_results = self.calculate_overall_quality_score(use_dynamic_weights=False)
        
        # Calculate with dynamic weights
        dynamic_results = self.calculate_overall_quality_score(use_dynamic_weights=True)
        
        # Print comparison
        print("\nOVERALL QUALITY SCORES:")
        print(f"   Static Weights:  {static_results['overall_score']:.1%}")
        print(f"   Dynamic Weights: {dynamic_results['overall_score']:.1%}")
        
        improvement = dynamic_results['overall_score'] - static_results['overall_score']
        improvement_pct = (improvement / static_results['overall_score'] * 100 
                        if static_results['overall_score'] > 0 else 0)
        
        print(f"\nIMPROVEMENT WITH DYNAMIC WEIGHTS: {improvement_pct:+.1f}%")
        
        # Weight comparison
        print("\nAVERAGE WEIGHTS USED:")
        print(f"   Static:  Skill={static_results['weights_used']['skill']:.0%}, "
            f"Schedule={static_results['weights_used']['schedule']:.0%}, "
            f"Travel={static_results['weights_used']['travel']:.0%}")
        print(f"   Dynamic: Skill={dynamic_results['weights_used']['skill']:.0%}, "
            f"Schedule={dynamic_results['weights_used']['schedule']:.0%}, "
            f"Travel={dynamic_results['weights_used']['travel']:.0%}")
        
        print("\n" + "="*80)
