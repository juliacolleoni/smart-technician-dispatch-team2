# 🚀 Smart Technician Dispatch - TELUS Hackathon Solution

**TELUS CAIO B2B Hackathon #1** | December 29-31, 2025

---

## ⚡ Quick Start (3 Steps)

```bash
# Step 1: Install dependencies
pip3 install -r requirements.txt

# Step 2: Run the optimizer
python3 technician_dispatch_optimizer.py

# Step 3: Generate visualizations
python3 route_visualizer.py
```

**Done!** Open `map_after_interactive.html` in your browser.

---

## 📖 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Solution Overview](#-solution-overview)
3. [How It Works](#-how-it-works)
4. [Installation & Usage](#-installation--usage)
5. [Results & Metrics](#-results--metrics)
6. [File Structure](#-file-structure)

---

## 🎯 Problem Statement

Optimize technician scheduling for **198 work orders** across **10 technicians** for one week in Calgary, balancing three competing priorities:

1. ✅ **Skill Matching** - Right technician for the job type
2. ✅ **Calendar Availability** - Respecting existing schedules  
3. ✅ **Route Efficiency** - Minimizing travel distance

### Hard Constraints
- Shift hours: 08:00-18:00 only
- No overlapping assignments
- Respect all job durations exactly
- Only schedule in AVAILABLE blocks (no UNAVAILABLE overlaps)

---

## 💡 Solution Overview

### Three-Pillar Scoring System

**1. Skill Matching (35% weight)**
- NLP keyword extraction from work order descriptions
- 12 skill categories: fiber, internet, TV, repair, install, troubleshoot, etc.
- Matches with technician skill profiles
- Bonus for previous service history with high satisfaction

**2. Availability Optimization (30% weight)**
- Respects existing calendar blocks (breaks, meetings, PTO, training)
- Matches customer time preferences (morning/afternoon)
- Prevents schedule conflicts and overlaps
- Minimizes schedule fragmentation

**3. Route Optimization (35% weight)**
- Haversine distance calculation between job locations (6371km Earth radius)
- Nearest-neighbor algorithm for route sequencing
- Minimizes travel distance between consecutive jobs
- Reduces backtracking across the city

### Algorithm Approach

```
PHASE 1: Initial Assignment (Day-by-Day + Route Optimization)
  For each day:
    For each unassigned work order:
      1. Extract required skills from job description
      2. For each available technician:
         a. Calculate skill match score (0-1)
         b. Find best available time slot
         c. Compute travel score based on distance from last job
         d. Combine scores: (0.35×skill + 0.30×avail + 0.35×travel)
      3. Assign to highest-scoring technician
      4. Sequence jobs using nearest-neighbor routing
      5. Update availability tracking

PHASE 2: Overflow Reassignment (Three-Tier Strategy)
  For unassigned work orders:
    Tier 1: Try same week day with all technicians
    Tier 2: Try any day in week with all technicians
    Tier 3: Force-fit into any 15-minute gap while respecting constraints
```

---

## 🔧 How It Works

### Key Components

**1. Skill Extractor** (`technician_dispatch_optimizer.py` lines 27-65)
- Keyword-based NLP extraction
- Categories: fiber, router, modem, cable, internet, phone, TV, install, repair, upgrade, troubleshoot, config
- Returns skill match ratio (matched keywords / total keywords)

**2. Distance Calculator** (`technician_dispatch_optimizer.py` lines 115-145)
- Pre-computes 39,402 distance pairs
- Haversine formula: accounts for Earth's curvature
- Caches results for performance

**3. Availability Manager** (`technician_dispatch_optimizer.py` lines 147-260)
- Tracks scheduled_times: `{tech_id: {date: [(start, end)]}}`
- Checks UNAVAILABLE blocks from calendar
- Prevents overlapping assignments
- Validates shift hours (08:00-18:00)

**4. Route Sequencer** (`technician_dispatch_optimizer.py` lines 445-475)
- Nearest-neighbor algorithm for each technician's daily route
- Starts from technician's home location
- Selects closest unvisited job iteratively
- Updates job start/end times based on sequence

**5. Overflow Reassignment** (`technician_dispatch_optimizer.py` lines 477-580)
- Three-tier exhaustive search
- Sorts jobs by duration (shorter first for better bin-packing)
- Force-fits into 15-minute gaps when needed
- Maintains 100% constraint compliance

---

## 📦 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip3 install -r requirements.txt
```

**Packages installed:**
- pandas 2.3.3 - Data manipulation
- numpy 2.0.2 - Numerical operations
- openpyxl 3.1.5 - Excel file reading
- folium 0.20.0 - Interactive map generation

### Running the Solution

**Step 1: Run the Optimizer**
```bash
python3 technician_dispatch_optimizer.py
```

**Output:** `final_schedule.csv` (198 assignments)

**Step 2: Generate Route Maps**
```bash
python3 route_visualizer.py
```

**Output:** 
- `map_before_interactive.html` - Baseline (random assignment)
- `map_after_interactive.html` - Optimized routes with filters

### Using the Interactive Maps

1. Open either HTML file in your browser
2. Use Layer Control (top-right corner)
3. Toggle days: Click "📅 Monday - All Techs" to show/hide all technicians for that day
4. Each technician has a unique color
5. Click markers for job details (ID, time, duration, type)
6. Routes show: Home → Jobs → Home

---

## 📊 Results & Metrics

### Assignment Success
- ✅ **198/198 work orders assigned** (100%)
- ✅ **0 constraint violations**
- ✅ **All 4 hard constraints satisfied**

### Constraint Compliance
1. ✅ **Shift Hours (08:00-18:00)**: 0 jobs before 08:00, 0 after 18:00
2. ✅ **No Overlaps**: 0 overlapping assignments
3. ✅ **Job Durations**: All preserved exactly as specified
4. ✅ **UNAVAILABLE Blocks**: 0 conflicts with unavailable times

### Route Optimization
- Before optimization (random): 2,632 km total travel
- After optimization: 2,710 km total travel
- Note: Slight increase due to prioritizing skill matching and availability over pure distance minimization

### Score Distribution
- Average skill match: 0.67 (67% keyword overlap)
- Average availability score: 0.94 (94% optimal time slot fit)
- Average travel score: 0.82 (67% jobs closely clustered)
- Average composite score: 0.81 (81% overall optimization)

### Technician Utilization
- T-01: 20 jobs
- T-02: 19 jobs
- T-03: 21 jobs
- T-04: 20 jobs
- T-05: 19 jobs
- T-06: 20 jobs
- T-07: 20 jobs
- T-08: 19 jobs
- T-09: 20 jobs
- T-10: 20 jobs

**Balance:** Max deviation = 2 jobs (10% variance) ✅

---

## 📁 File Structure

### Core Solution Files

```
technician_dispatch_optimizer.py (20 KB, 619 lines)
├── SkillExtractor          - NLP keyword extraction
├── DataLoader              - Excel file processing
├── DistanceCalculator      - Haversine distance matrix
├── AvailabilityManager     - Schedule conflict tracking
└── TechnicianDispatchOptimizer
    ├── optimize_schedule_by_day_and_route()  - Main optimization
    ├── _sequence_jobs_by_distance()          - Nearest-neighbor routing
    ├── _reassign_overflow_jobs()             - Three-tier reassignment
    └── _make_assignment()                    - Assignment helper

route_visualizer.py (10 KB, 380 lines)
├── RouteVisualizer
│   ├── create_interactive_filtered_map()  - Day-grouped maps
│   ├── create_map()                       - Individual route map
│   └── calculate_total_distance()         - Travel metrics
```

### Input Data (Provided)
- `01_technician_profiles.xlsx` - 10 technicians, skills, locations
- `02_availability_schedules.xlsx` - Calendar blocks (available/unavailable)
- `03_customer_history.xlsx` - Previous service records
- `04_workorders_week_original.xlsx` - 198 work orders to schedule
- `05_distances.xlsx` - Pre-computed distance matrix (optional)
- `06_locations_nodes.xlsx` - Geographic coordinates

### Output Files (Generated)
- `final_schedule.csv` - Optimized schedule (198 rows)
- `map_before_interactive.html` - Baseline visualization (536 KB)
- `map_after_interactive.html` - Optimized visualization (488 KB)

### Documentation
- `README.md` - This comprehensive guide
- `requirements.txt` - Python dependencies

---
