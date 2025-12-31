# Solution Concept Diagram

## Architecture Overview

```mermaid
flowchart TB
    subgraph INPUT["📥 INPUT DATA"]
        T[01_technician_profiles.xlsx<br/>Skills, Location, Capacity]
        C[02_customer_profiles.xlsx<br/>Customer Info]
        H[03_customer_service_history.xlsx<br/>Past Performance]
        W[04_workorders_week_original.xlsx<br/>Jobs + Coordinates]
        A[05_technician_calendar_original.xlsx<br/>Availability Blocks]
    end

    subgraph LOAD["🔄 DATA LOADING"]
        DL[DataLoader<br/>Load & Validate]
    end

    subgraph PREP["⚙️ PREPROCESSING"]
        SE[SkillExtractor<br/>NLP Keyword Matching]
        DC[DistanceCalculator<br/>Haversine Matrix]
        AM[AvailabilityManager<br/>Calendar Parsing]
        DW[DynamicWeightManager<br/>Context-Based Scoring]
    end

    subgraph OPT["🎯 OPTIMIZATION ENGINE"]
        P0[Phase 0:<br/>Territorial Distribution<br/>Spatial Load Balancing]
        P1[Phase 1:<br/>Greedy Assignment<br/>Composite Scoring]
        P2[Phase 2:<br/>Overflow Reassignment<br/>Force-Fit Remaining]
    end

    subgraph SCORE["📊 SCORING SYSTEM"]
        S1[Skill Match<br/>35% weight]
        S2[Availability<br/>30% weight]
        S3[Travel Distance<br/>35% weight]
        CS[Composite Score<br/>Dynamic Weighting]
    end

    subgraph ROUTE["🗺️ ROUTE OPTIMIZATION"]
        NN[Nearest Neighbor<br/>Sequential Routing]
        SEQ[Time Sequencing<br/>No Overlaps]
    end

    subgraph OUTPUT["📤 OUTPUT GENERATION"]
        CSV[final_schedule.csv<br/>191 Assignments]
        MAP1[map_before_interactive.html<br/>Baseline Visualization]
        MAP2[map_after_interactive.html<br/>Optimized Visualization]
        SCHED[technician_schedules_detailed.xlsx<br/>Individual Schedules]
    end

    subgraph METRICS["📈 QUALITY METRICS"]
        M1[Skill Match Score<br/>+82.9% improvement]
        M2[Schedule Preference<br/>+27.2% improvement]
        M3[Travel Efficiency<br/>173km saved]
        M4[Overall Quality<br/>+31.5% improvement]
        M5[Reassignment Rate<br/>89.5% changed]
    end

    INPUT --> LOAD
    LOAD --> PREP
    
    T --> SE
    W --> DC
    A --> AM
    
    PREP --> OPT
    
    OPT --> P0
    P0 --> P1
    P1 --> P2
    
    P1 --> SCORE
    S1 --> CS
    S2 --> CS
    S3 --> CS
    DW --> CS
    
    CS --> ROUTE
    ROUTE --> NN
    NN --> SEQ
    
    SEQ --> OUTPUT
    
    OUTPUT --> METRICS
    
    CSV --> M1
    CSV --> M2
    CSV --> M3
    CSV --> M4
    CSV --> M5

    style INPUT fill:#e1f5ff
    style PREP fill:#fff4e1
    style OPT fill:#f0e1ff
    style SCORE fill:#e1ffe1
    style OUTPUT fill:#ffe1e1
    style METRICS fill:#ffe1f5
```

## Component Details

### 1️⃣ Data Loading Layer
- **DataLoader**: Centralized Excel file reading
- **Validation**: Data type checking and normalization
- **Preprocessing**: Date parsing, coordinate extraction

### 2️⃣ Analysis Components
- **SkillExtractor**: NLP-based keyword extraction from job descriptions
- **DistanceCalculator**: Pre-computes 39,006 pairwise distances using haversine formula
- **AvailabilityManager**: Tracks calendar blocks and prevents conflicts
- **DynamicWeightManager**: Adjusts scoring weights based on job context

### 3️⃣ Three-Phase Optimization
```
Phase 0: TERRITORIAL DISTRIBUTION
├── Maximize spatial separation between technicians
└── Assign first job to each tech for day-level load balancing

Phase 1: GREEDY ASSIGNMENT
├── For each unassigned job:
│   ├── Score all available technicians
│   ├── Consider: skills × availability × travel
│   └── Assign to highest scorer
└── Use nearest-neighbor for route sequencing

Phase 2: OVERFLOW REASSIGNMENT
├── Handle jobs that couldn't fit in Phase 1
├── Try same day with all techs
├── Try any day in week
└── Force-fit with constraint relaxation (flexible mode)
```

### 4️⃣ Scoring Algorithm
```python
composite_score = (
    skill_weight × skill_match_score +
    availability_weight × time_slot_score +
    travel_weight × distance_efficiency_score
)
```

Dynamic weights adapt based on:
- **Job type**: Installation, repair, upgrade, troubleshoot
- **Customer tier**: Standard, premium, business
- **Time sensitivity**: Low, medium, high urgency

### 5️⃣ Output Generation
- **CSV Schedule**: 191 optimized assignments with rationale
- **Interactive Maps**: Folium-based visualization with tech/day filtering
- **Individual Schedules**: Per-technician Excel workbook with calendar integration
- **Quality Reports**: Comprehensive metrics and comparison analysis

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Optimizer
    participant DataLoader
    participant Calculator
    participant AvailMgr
    participant Output

    User->>Optimizer: Run optimization
    Optimizer->>DataLoader: Load 5 Excel files
    DataLoader-->>Optimizer: Return datasets
    
    Optimizer->>Calculator: Build distance matrix
    Calculator-->>Optimizer: 39,006 distances cached
    
    Optimizer->>AvailMgr: Build availability map
    AvailMgr-->>Optimizer: Calendar indexed
    
    loop For each date
        Optimizer->>Optimizer: Phase 0: Territorial assignment
        loop For each job
            Optimizer->>Calculator: Score technicians
            Calculator-->>Optimizer: Composite scores
            Optimizer->>AvailMgr: Check availability
            AvailMgr-->>Optimizer: Time slots
            Optimizer->>Optimizer: Assign to best match
        end
        Optimizer->>Optimizer: Phase 1: Route optimization
    end
    
    Optimizer->>Optimizer: Phase 2: Overflow handling
    
    Optimizer->>Output: Generate schedule CSV
    Optimizer->>Output: Generate HTML maps
    Optimizer->>Output: Calculate metrics
    
    Output-->>User: Results ready
```

## Technology Stack

```mermaid
graph LR
    subgraph Core
        PY[Python 3.9]
    end
    
    subgraph Data
        PD[pandas<br/>DataFrames]
        NP[numpy<br/>Numerics]
        XL[openpyxl<br/>Excel I/O]
    end
    
    subgraph Geo
        HAV[Haversine<br/>Distance Calc]
    end
    
    subgraph Viz
        FOL[folium<br/>Interactive Maps]
        HTML[HTML Export]
    end
    
    subgraph Algorithms
        GRD[Greedy Heuristic]
        NN[Nearest Neighbor]
        DYN[Dynamic Weighting]
    end
    
    PY --> Data
    PY --> Geo
    PY --> Viz
    PY --> Algorithms
```

## Key Metrics & Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Skill Match** | 36.6% | 67.0% | **+82.9%** ✓ |
| **Schedule Preference** | 54.1% | 68.8% | **+27.2%** ✓ |
| **Travel Efficiency** | 75.7% | 79.7% | **+5.2%** ✓ |
| **Total Distance** | 2,927 km | 2,753 km | **-173 km** ✓ |
| **Overall Quality** | 54.3% | 71.5% | **+31.5%** ✓ |
| **Reassignment Rate** | - | 89.5% | 171/191 jobs |

## Constraints & Assumptions

### Hard Constraints ⛔
- Shift hours: 08:00-18:00 only
- No overlapping assignments
- Respect all job durations
- Only schedule in AVAILABLE blocks
- Max jobs per day limits (flexible +1 in overflow mode)

### Soft Constraints ⚠️
- Customer time preferences (bonused in scoring)
- Minimize travel distance
- Prefer skilled technicians
- Balance workload across team

### Assumptions 📋
- Straight-line distance (haversine) approximates road distance
- Job durations are accurate
- No traffic or weather delays
- Technicians start from home base
- No multi-technician jobs
- Calendar events are immutable

---

**Generated by**: Smart Technician Dispatch Optimizer  
**TELUS CAIO B2B Hackathon #1** | December 29-31, 2025
