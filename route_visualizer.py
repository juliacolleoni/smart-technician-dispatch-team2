"""
Route Visualization for Technician Dispatch
Creates before/after maps showing route optimization
"""

import pandas as pd
import folium
from folium import plugins
import numpy as np
from datetime import datetime

# Color palette for different technicians
TECH_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
]

class RouteVisualizer:
    """Create interactive maps showing technician routes."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """Load necessary datasets."""
        print("Loading data for visualization...")
        
        self.work_orders = pd.read_excel(f"{self.data_dir}/04_workorders_week_original.xlsx")
        self.technicians = pd.read_excel(f"{self.data_dir}/01_technician_profiles.xlsx")
        
        try:
            self.final_schedule = pd.read_csv("output/final_schedule.csv")
        except:
            print("⚠ final_schedule.csv not found. Run optimizer first.")
            self.final_schedule = None
        
        print("✓ Data loaded")
    
    def create_map(self, schedule_df: pd.DataFrame, tech_col: str, 
                   title: str, output_file: str, date_filter: str = None):
        """
        Create a map showing routes for all technicians.
        
        Args:
            schedule_df: DataFrame with schedule
            tech_col: Column name for technician ID
            title: Map title
            output_file: Output HTML file path
            date_filter: Optional date to filter (YYYY-MM-DD)
        """
        print(f"\nCreating map: {title}")
        
        # Filter by date if specified
        if date_filter:
            schedule_df = schedule_df[schedule_df['optimized_scheduled_date'] == date_filter].copy()
        
        # Center map on Calgary
        calgary_center = [51.0447, -114.0719]
        m = folium.Map(location=calgary_center, zoom_start=11, tiles='OpenStreetMap')
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 50px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:16px; padding: 10px">
            <b>{title}</b>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Group by technician
        techs = schedule_df[tech_col].unique()
        
        for i, tech_id in enumerate(techs):
            if pd.isna(tech_id):
                continue
            
            tech_jobs = schedule_df[schedule_df[tech_col] == tech_id].copy()
            
            # Sort by time to get route order
            if 'optimized_start_time' in tech_jobs.columns:
                tech_jobs['start_minutes'] = tech_jobs['optimized_start_time'].apply(
                    lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) else 0
                )
                tech_jobs = tech_jobs.sort_values('start_minutes')
            
            # Get locations
            locations = []
            for _, job in tech_jobs.iterrows():
                wo = self.work_orders[self.work_orders['workorder_id'] == job['workorder_id']]
                if not wo.empty:
                    lat = wo.iloc[0]['job_lat']
                    lon = wo.iloc[0]['job_lon']
                    locations.append({
                        'lat': lat,
                        'lon': lon,
                        'workorder_id': job['workorder_id'],
                        'time': job.get('optimized_start_time', 'N/A')
                    })
            
            if not locations:
                continue
            
            color = TECH_COLORS[i % len(TECH_COLORS)]
            
            # Draw route line
            route_coords = [(loc['lat'], loc['lon']) for loc in locations]
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Technician: {tech_id} - {len(locations)} jobs",
                tooltip=folium.Tooltip(f"{tech_id} - {len(locations)} visits", permanent=False, sticky=True)
            ).add_to(m)
            
            # Add markers for each job
            for j, loc in enumerate(locations):
                folium.CircleMarker(
                    location=[loc['lat'], loc['lon']],
                    radius=8,
                    popup=f"<b>{tech_id}</b><br>{loc['workorder_id']}<br>Time: {loc['time']}<br>Stop #{j+1}",
                    tooltip=folium.Tooltip(f"#{j+1}: {loc['workorder_id']} - {loc['time']}", permanent=False, sticky=True),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.9
                ).add_to(m)
            
            # Add start marker (larger)
            folium.Marker(
                location=[locations[0]['lat'], locations[0]['lon']],
                popup=f"<b>{tech_id} - Start</b><br>{locations[0]['workorder_id']}",
                tooltip=folium.Tooltip(f"{tech_id} - Start", permanent=False, sticky=True),
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
            <p><b>Legend</b></p>
        '''
        for i, tech_id in enumerate(list(techs)[:10]):
            if pd.notna(tech_id):
                color = TECH_COLORS[i % len(TECH_COLORS)]
                legend_html += f'<p><span style="color:{color}">●</span> {tech_id}</p>'
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(output_file)
        print(f"✓ Saved map to {output_file}")
        
        return m
    
    def calculate_total_distance(self, schedule_df: pd.DataFrame, tech_col: str) -> float:
        """Calculate total travel distance for all technicians."""
        from technician_dispatch_optimizer import haversine_distance
        
        total_distance = 0
        techs = schedule_df[tech_col].unique()
        
        for tech_id in techs:
            if pd.isna(tech_id):
                continue
            
            tech_jobs = schedule_df[schedule_df[tech_col] == tech_id].copy()
            
            # Sort by time
            if 'optimized_start_time' in tech_jobs.columns:
                tech_jobs['start_minutes'] = tech_jobs['optimized_start_time'].apply(
                    lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1]) if pd.notna(x) else 0
                )
                tech_jobs = tech_jobs.sort_values('start_minutes')
            
            # Calculate distance between consecutive jobs
            prev_lat, prev_lon = None, None
            for _, job in tech_jobs.iterrows():
                wo = self.work_orders[self.work_orders['workorder_id'] == job['workorder_id']]
                if not wo.empty:
                    lat = wo.iloc[0]['job_lat']
                    lon = wo.iloc[0]['job_lon']
                    
                    if prev_lat is not None:
                        dist = haversine_distance(prev_lat, prev_lon, lat, lon)
                        total_distance += dist
                    
                    prev_lat, prev_lon = lat, lon
        
        return total_distance
    
    def create_interactive_filtered_map(self, schedule_data, tech_id_column, title, output_file):
        """Create an interactive map with filters for technician and day of week."""
        print(f"\nCreating map: {title}")
        
        # Merge with work orders to get coordinates
        schedule_with_coords = schedule_data.merge(
            self.work_orders[['workorder_id', 'job_lat', 'job_lon', 'neighborhood', 'job_type', 'job_duration_minutes']],
            on='workorder_id',
            how='left'
        )
        
        # Parse rationale to extract scores
        def parse_rationale(rationale_str):
            """Extract skill, availability, and travel scores from rationale string."""
            if pd.isna(rationale_str):
                return None, None, None, None, None, None, None
            
            try:
                # Format: "Skill:0.50×0.38 Avail:1.00×0.33 Travel:0.50×0.29 | Total:0.67"
                parts = str(rationale_str).split(' ')
                
                skill_score = None
                skill_weight = None
                avail_score = None
                avail_weight = None
                travel_score = None
                travel_weight = None
                total_score = None
                
                for part in parts:
                    if part.startswith('Skill:'):
                        values = part.split(':')[1].split('×')
                        skill_score = float(values[0])
                        skill_weight = float(values[1])
                    elif part.startswith('Avail:'):
                        values = part.split(':')[1].split('×')
                        avail_score = float(values[0])
                        avail_weight = float(values[1])
                    elif part.startswith('Travel:'):
                        values = part.split(':')[1].split('×')
                        travel_score = float(values[0])
                        travel_weight = float(values[1])
                    elif part.startswith('Total:'):
                        total_score = float(part.split(':')[1])
                
                return skill_score, skill_weight, avail_score, avail_weight, travel_score, travel_weight, total_score
            except:
                return None, None, None, None, None, None, None
        
        # Add score columns
        schedule_with_coords[['skill_score', 'skill_weight', 'avail_score', 'avail_weight', 
                              'travel_score', 'travel_weight', 'total_score']] = schedule_with_coords['rationale'].apply(
            lambda x: pd.Series(parse_rationale(x))
        )
        
        # Normalize date format - convert to date only (remove time component)
        # Handle mixed formats (some with timestamps, some without)
        schedule_with_coords['optimized_scheduled_date'] = pd.to_datetime(
            schedule_with_coords['optimized_scheduled_date'], format='mixed', errors='coerce'
        ).dt.date
        
        # Get center point
        avg_lat = schedule_with_coords['job_lat'].mean()
        avg_lon = schedule_with_coords['job_lon'].mean()
        
        # Create base map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles='OpenStreetMap')
        
        # Get unique dates and technicians
        all_dates = sorted(schedule_with_coords['optimized_scheduled_date'].unique())
        all_techs = sorted(schedule_with_coords[tech_id_column].unique())
        
        # Create consistent color mapping for all technicians
        tech_colors = {
            'T-01': '#FF6B6B',  # Red
            'T-02': '#4ECDC4',  # Turquoise
            'T-03': '#45B7D1',  # Light Blue
            'T-04': '#FFA07A',  # Light Salmon
            'T-05': '#98D8C8',  # Mint
            'T-06': '#F7DC6F',  # Yellow
            'T-07': '#BB8FCE',  # Purple
            'T-08': '#85C1E2',  # Sky Blue
            'T-09': '#52B788',  # Green
            'T-10': '#F8B739',  # Orange
        }
        
        # Create feature groups organized by day (so all techs for a day can be toggled together)
        for date in all_dates:
            date_str = str(date)
            day_name = pd.to_datetime(date).strftime('%A')
            
            # Create a parent feature group for this day
            day_group = folium.FeatureGroup(name=f'{day_name} ({date_str}) - All Techs', show=False)
            
            for tech_id in all_techs:
                # Filter data for this tech and day
                tech_day_data = schedule_with_coords[
                    (schedule_with_coords[tech_id_column] == tech_id) & 
                    (schedule_with_coords['optimized_scheduled_date'] == date)
                ].copy()
                
                if len(tech_day_data) == 0:
                    continue
                
                # Get tech location
                tech_info = self.technicians[self.technicians['technician_id'] == tech_id].iloc[0]
                tech_lat = tech_info['home_base_lat']
                tech_lon = tech_info['home_base_lon']
                
                # Add tech home marker
                folium.Marker(
                    [tech_lat, tech_lon],
                    popup=f"<b>{tech_id} Home</b>",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa'),
                    tooltip=folium.Tooltip(f"{tech_id} - Home Base", permanent=False, sticky=True)
                ).add_to(day_group)
                
                # Sort jobs by time
                tech_day_data = tech_day_data.sort_values('optimized_start_time')
                
                # Get consistent color for this technician
                route_color = tech_colors.get(tech_id, '#808080')  # Gray as fallback
                
                # Add job markers and route
                route_coords = [(tech_lat, tech_lon)]
                
                for visit_num, (idx, job) in enumerate(tech_day_data.iterrows(), start=1):
                    job_lat = job['job_lat']
                    job_lon = job['job_lon']
                    route_coords.append((job_lat, job_lon))
                    
                    # Preparar informações dos scores
                    skill_score = job.get('skill_score', 0)
                    skill_weight = job.get('skill_weight', 0)
                    avail_score = job.get('avail_score', 0)
                    avail_weight = job.get('avail_weight', 0)
                    travel_score = job.get('travel_score', 0)
                    travel_weight = job.get('travel_weight', 0)
                    total_score = job.get('total_score', 0)
                    
                    # Função para criar barra de progresso visual
                    def get_progress_bar(score, color):
                        if pd.isna(score):
                            return ""
                        percentage = score * 100
                        return f'''
                        <div style="
                            background-color: #e0e0e0;
                            border-radius: 10px;
                            height: 10px;
                            width: 100%;
                            overflow: hidden;
                            margin-top: 2px;
                        ">
                            <div style="
                                background-color: {color};
                                height: 100%;
                                width: {percentage}%;
                                border-radius: 10px;
                            "></div>
                        </div>
                        '''
                    
                    # Criar seção de scores para o tooltip
                    scores_html = ""
                    if not pd.isna(skill_score):
                        scores_html = f'''
                        <div style="
                            margin-top: 10px;
                            padding-top: 10px;
                            border-top: 2px solid #e0e0e0;
                        ">
                            <div style="font-weight: bold; margin-bottom: 8px; color: #333;">
                                Scores de Otimização
                            </div>
                            
                            <div style="margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span><strong>Skills:</strong></span>
                                    <span>{skill_score:.2f} × {skill_weight:.2f}</span>
                                </div>
                                {get_progress_bar(skill_score, '#4CAF50')}
                            </div>
                            
                            <div style="margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span><strong>Schedule:</strong></span>
                                    <span>{avail_score:.2f} × {avail_weight:.2f}</span>
                                </div>
                                {get_progress_bar(avail_score, '#2196F3')}
                            </div>
                            
                            <div style="margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span><strong>Distância:</strong></span>
                                    <span>{travel_score:.2f} × {travel_weight:.2f}</span>
                                </div>
                                {get_progress_bar(travel_score, '#FF9800')}
                            </div>
                            
                            <div style="
                                margin-top: 8px;
                                padding-top: 6px;
                                border-top: 1px solid #e0e0e0;
                                font-weight: bold;
                                color: {route_color};
                            ">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Total Score:</span>
                                    <span>{total_score:.2f}</span>
                                </div>
                            </div>
                        </div>
                        '''
                    
                    popup_text = f"""
                    <b>Job: {job['workorder_id']}</b><br>
                    Tech: {job[tech_id_column]}<br>
                    Visit #{visit_num}<br>
                    Time: {job['optimized_start_time']} - {job['optimized_end_time']}<br>
                    Duration: {job['job_duration_minutes']} min<br>
                    Type: {job['job_type']}<br>
                    Location: {job['neighborhood']}
                    {scores_html}
                    """
                    
                    # Create tooltip with score information
                    tooltip_html = f"""
                    <div style="font-family: Arial, sans-serif; font-size: 11px; min-width: 200px;">
                        <div style="font-weight: bold; margin-bottom: 5px; font-size: 12px;">
                            #{visit_num}: {job['workorder_id']}
                        </div>
                        <div style="margin-bottom: 3px;">
                            ⏰ {job['optimized_start_time']} - {job['optimized_end_time']}
                        </div>
                        <div style="margin-bottom: 3px;">
                            📍 {job['neighborhood']} | {job['job_type']}
                        </div>
                    """
                    
                    # Add scores if available
                    if not pd.isna(skill_score):
                        tooltip_html += f"""
                        <div style="border-top: 1px solid #ccc; margin-top: 5px; padding-top: 5px;">
                            <div style="margin-bottom: 2px;">
                                🎯 Skill: {skill_score:.2f} × {skill_weight:.2f}
                            </div>
                            <div style="margin-bottom: 2px;">
                                📅 Schedule: {avail_score:.2f} × {avail_weight:.2f}
                            </div>
                            <div style="margin-bottom: 2px;">
                                🚗 Distance: {travel_score:.2f} × {travel_weight:.2f}
                            </div>
                            <div style="font-weight: bold; color: {route_color}; margin-top: 3px;">
                                ⭐ Total: {total_score:.2f}
                            </div>
                        </div>
                        """
                    
                    tooltip_html += "</div>"
                    
                    # Create numbered marker
                    folium.Marker(
                        [job_lat, job_lon],
                        popup=folium.Popup(popup_text, max_width=350),
                        icon=folium.DivIcon(html=f'''
                            <div style="
                                font-size: 12px;
                                font-weight: bold;
                                color: white;
                                text-align: center;
                                line-height: 24px;
                                width: 24px;
                                height: 24px;
                                border-radius: 50%;
                                background-color: {route_color};
                                border: 2px solid white;
                                box-shadow: 0 0 4px rgba(0,0,0,0.5);
                            ">{visit_num}</div>
                        '''),
                        tooltip=folium.Tooltip(tooltip_html, permanent=False, sticky=True)
                    ).add_to(day_group)
                
                # Add route back to home
                route_coords.append((tech_lat, tech_lon))
                
                # Draw route line
                folium.PolyLine(
                    route_coords,
                    color=route_color,
                    weight=2,
                    opacity=0.6,
                    popup=f"{tech_id} route - {len(tech_day_data)} jobs",
                    tooltip=folium.Tooltip(f"{tech_id} - {len(tech_day_data)} visits", permanent=False, sticky=True)
                ).add_to(day_group)
            
            # Add the day group to the map
            day_group.add_to(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Save map
        m.save(output_file)
        print(f"✓ Saved map to {output_file}")

    def create_comparison_report(self):
        """Create before/after comparison maps with filters."""
        if self.final_schedule is None:
            print("Cannot create comparison - final_schedule.csv not found")
            return
        
        print("\n" + "="*80)
        print("CREATING INTERACTIVE FILTERED ROUTE VISUALIZATIONS")
        print("="*80)
        
        # Use original assignments from dataset 04
        print("\nUsing original assignments from dataset for comparison...")
        original_schedule = self.final_schedule.copy()
        
        # Verify that original_assigned_technician_id exists
        if 'original_assigned_technician_id' not in original_schedule.columns:
            print("⚠ Warning: original_assigned_technician_id not found in final_schedule.csv")
            print("Using optimized assignments for both maps...")
        
        # Create interactive filtered map for BEFORE (Original Assignments)
        print("\nCreating interactive BEFORE map (original assignments from dataset)...")
        self.create_interactive_filtered_map(
            original_schedule,
            'original_assigned_technician_id',
            'Before Optimization (Original Assignments)',
            "output/map_before_interactive.html"
        )
        
        # Create interactive filtered map for AFTER
        print("\nCreating interactive AFTER map (filter by technician and day)...")
        self.create_interactive_filtered_map(
            original_schedule,
            'optimized_assigned_technician_id',
            'After Optimization - Interactive Filter',
            "output/map_after_interactive.html"
        )
        
        # Calculate total distances
        total_before_distance = self.calculate_total_distance(original_schedule, 'original_assigned_technician_id')
        total_after_distance = self.calculate_total_distance(original_schedule, 'optimized_assigned_technician_id')
        
        distance_saved = total_before_distance - total_after_distance
        percent_saved = (distance_saved / total_before_distance * 100) if total_before_distance > 0 else 0
        
        # Print summary
        print("\n" + "="*80)
        print(f"ROUTE OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"Before (original assignments): {total_before_distance:.2f} km total travel")
        print(f"After (optimized assignments): {total_after_distance:.2f} km total travel")
        
        if distance_saved > 0:
            print(f"Distance saved: {distance_saved:.2f} km ({percent_saved:.1f}% reduction)")
        elif distance_saved < 0:
            print(f"Distance increased: {abs(distance_saved):.2f} km ({abs(percent_saved):.1f}% increase)")
            print("Note: Optimization prioritized skill match and schedule preferences over pure distance")
        else:
            print(f"Distance unchanged: {total_after_distance:.2f} km")
        
        print("\n✓ Visualization complete!")
        print(f"\nGenerated interactive maps:")
        print(f"   - output/map_before_interactive.html (original assignments from dataset)")
        print(f"   - output/map_after_interactive.html (optimized assignments)")
        print(f"\nOpen these files and use the layer control (top-right) to filter by day!")
        print(f"   Each map shows all technicians for each day - toggle days to compare.")
        print(f"   Compare BEFORE (original) vs AFTER (optimized) to see the improvement!")

def main():
    """Main execution."""
    data_dir = "data"
    
    visualizer = RouteVisualizer(data_dir)
    visualizer.create_comparison_report()

if __name__ == "__main__":
    main()
