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
        self.locations = pd.read_excel(f"{self.data_dir}/06_locations_nodes.xlsx")
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
                popup=f"Technician: {tech_id}"
            ).add_to(m)
            
            # Add markers for each job
            for j, loc in enumerate(locations):
                folium.CircleMarker(
                    location=[loc['lat'], loc['lon']],
                    radius=8,
                    popup=f"<b>{tech_id}</b><br>{loc['workorder_id']}<br>Time: {loc['time']}<br>Stop #{j+1}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.9
                ).add_to(m)
            
            # Add start marker (larger)
            folium.Marker(
                location=[locations[0]['lat'], locations[0]['lon']],
                popup=f"<b>{tech_id} - Start</b><br>{locations[0]['workorder_id']}",
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
        
        # Get center point
        avg_lat = schedule_with_coords['job_lat'].mean()
        avg_lon = schedule_with_coords['job_lon'].mean()
        
        # Create base map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles='OpenStreetMap')
        
        # Get unique dates and technicians
        all_dates = sorted(schedule_with_coords['optimized_scheduled_date'].unique())
        all_techs = sorted(schedule_with_coords[tech_id_column].unique())
        
        # Create feature groups organized by day (so all techs for a day can be toggled together)
        for date in all_dates:
            date_str = str(date).split()[0]
            day_name = pd.to_datetime(date).strftime('%A')
            
            # Create a parent feature group for this day
            day_group = folium.FeatureGroup(name=f'📅 {day_name} ({date_str}) - All Techs', show=False)
            
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
                    tooltip=f"{tech_id} Home"
                ).add_to(day_group)
                
                # Sort jobs by time
                tech_day_data = tech_day_data.sort_values('optimized_start_time')
                
                # Calculate color for this route
                tech_index = all_techs.index(tech_id)
                colors = ['red', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                         'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 
                         'pink', 'lightblue', 'lightgreen', 'gray', 'black']
                route_color = colors[tech_index % len(colors)]
                
                # Add job markers and route
                route_coords = [(tech_lat, tech_lon)]
                
                for idx, job in tech_day_data.iterrows():
                    job_lat = job['job_lat']
                    job_lon = job['job_lon']
                    route_coords.append((job_lat, job_lon))
                    
                    popup_text = f"""
                    <b>Job: {job['workorder_id']}</b><br>
                    Tech: {job[tech_id_column]}<br>
                    Time: {job['optimized_start_time']} - {job['optimized_end_time']}<br>
                    Duration: {job['job_duration_minutes']} min<br>
                    Type: {job['job_type']}<br>
                    Location: {job['neighborhood']}
                    """
                    
                    folium.CircleMarker(
                        [job_lat, job_lon],
                        radius=8,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=route_color,
                        fill=True,
                        fillColor=route_color,
                        fillOpacity=0.7,
                        tooltip=f"{job['workorder_id']} - {job['optimized_start_time']}"
                    ).add_to(day_group)
                
                # Add route back to home
                route_coords.append((tech_lat, tech_lon))
                
                # Draw route line
                folium.PolyLine(
                    route_coords,
                    color=route_color,
                    weight=2,
                    opacity=0.6,
                    popup=f"{tech_id} route"
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
        
        # Create a "before" schedule with random assignments for comparison
        print("\nGenerating baseline (unoptimized) schedule for comparison...")
        original_schedule = self.final_schedule.copy()
        
        # Create simulated "before" by randomly assigning technicians
        import random
        random.seed(42)  # For reproducibility
        
        available_techs = self.technicians['technician_id'].tolist()
        original_schedule['simulated_original_tech'] = [
            random.choice(available_techs) for _ in range(len(original_schedule))
        ]
        
        # Create interactive filtered map for BEFORE
        print("\nCreating interactive BEFORE map (filter by technician and day)...")
        self.create_interactive_filtered_map(
            original_schedule,
            'simulated_original_tech',
            'Before Optimization - Interactive Filter',
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
        total_before_distance = self.calculate_total_distance(original_schedule, 'simulated_original_tech')
        total_after_distance = self.calculate_total_distance(original_schedule, 'optimized_assigned_technician_id')
        
        distance_saved = total_before_distance - total_after_distance
        percent_saved = (distance_saved / total_before_distance * 100) if total_before_distance > 0 else 0
        
        # Print summary
        print("\n" + "="*80)
        print(f"ROUTE OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"Before (random assignment): {total_before_distance:.2f} km total travel")
        print(f"After (route optimized): {total_after_distance:.2f} km total travel")
        print(f"Distance saved: {distance_saved:.2f} km ({percent_saved:.1f}% reduction)")
        
        print("\n✓ Visualization complete!")
        print(f"\n📁 Generated interactive maps:")
        print(f"   - output/map_before_interactive.html (filterable by technician & day)")
        print(f"   - output/map_after_interactive.html (filterable by technician & day)")
        print(f"\n💡 Open these files and use the layer control (top-right) to filter by technician and day!")
        print(f"   Toggle different combinations to compare routes across different days and technicians.")

def main():
    """Main execution."""
    data_dir = "data"
    
    visualizer = RouteVisualizer(data_dir)
    visualizer.create_comparison_report()

if __name__ == "__main__":
    main()
