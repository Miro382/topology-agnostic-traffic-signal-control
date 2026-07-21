import random

VEHICLE_ROUTES = [
    "N1_in W1_out", "N1_in S1_out", "N1_in E1_out",
    "E1_in N1_out", "E1_in W1_out", "E1_in S1_out",
    "S1_in E1_out", "S1_in N1_out", "S1_in W1_out",
    "W1_in S1_out", "W1_in E1_out", "W1_in N1_out",
]

RAIL_ROUTES = [
    "ST_in NT_out", "NT_in ST_out",
]

PEDESTRIAN_ROUTES = [
    ("SW_in", "-E23"),
    ("W3_in", "E3_out"),
    ("SE_in", "E22"),
    ("S3_in", "N3_out"),
    ("NE_in", "E21"),
    ("E3_in", "W3_out"),
    ("NW_in", "-E24"),
]

PEDESTRIAN_ROUTES_CANONICAL = [
    ("PSE_in", "PSEI_in"),
    ("PSW_in", "PSWI_in"),
    ("PNE_in", "PNEI_in"),
    ("PNW_in", "PNWI_in"),
    ("PSEI_in", "PNEI_out"),
    ("PNEI_in", "PNWI_out"),
    ("PNWI_in", "PSWI_out"),
    ("PSWI_in", "PSEI_out"),
]


# Vehicle pairs whose movements never cross
NON_CONFLICTING_PAIRS = [
    ("N1_in S1_out", "S1_in N1_out"),  # straight through N <-> S
    ("E1_in W1_out", "W1_in E1_out"),  # straight through E <-> W
    ("N1_in W1_out", "S1_in E1_out"),  # right turn N + right turn S (both turn right, no conflict)
    ("E1_in N1_out", "W1_in S1_out"),  # right turn E + right turn W
    ("N1_in E1_out", "S1_in W1_out"),  # left turn N + left turn S (opposite, no crossing)
    ("E1_in S1_out", "W1_in N1_out"),  # left turn E + left turn W (opposite, no crossing)
    ("N1_in W1_out", "E1_in N1_out"),  # right turn N + right turn E
    ("S1_in E1_out", "W1_in S1_out"),  # right turn S + right turn W
    ("N1_in W1_out", "W1_in S1_out"),  # right turn N + right turn W
    ("S1_in E1_out", "E1_in N1_out"),  # right turn S + right turn E
    ("ST_in NT_out", "N1_in S1_out"),  # tram straight + car straight N<->S (tram on separate rail, no conflict)
    ("NT_in ST_out", "S1_in N1_out"),  # tram straight + car straight S<->N (tram on separate rail, no conflict)
    ("ST_in NT_out", "E1_in N1_out"),  # tram straight + car right turn E->N (tram on separate rail, no conflict)
]

BEST_CASE_COMBINATIONS = [
    ["N1_in W1_out", "N1_in S1_out", "NT_in ST_out", "ST_in NT_out", "S1_in E1_out", "S1_in N1_out"],
    ["N1_in W1_out", "N1_in S1_out", "S1_in E1_out", "S1_in N1_out"],
    ["E1_in N1_out", "E1_in W1_out", "W1_in S1_out", "W1_in E1_out"],
    ["N1_in W1_out", "N1_in S1_out", "N1_in E1_out", "E1_in N1_out"],
    ["E1_in N1_out", "E1_in W1_out", "E1_in S1_out", "S1_in E1_out"],
    ["S1_in E1_out", "S1_in N1_out", "N1_in W1_out", "W1_in S1_out"],
    ["W1_in S1_out", "W1_in E1_out", "W1_in N1_out", "N1_in W1_out"],
    ["N1_in W1_out", "E1_in N1_out", "S1_in E1_out", "W1_in S1_out", "ST_in NT_out", "NT_in ST_out"],
    ["N1_in W1_out", "E1_in N1_out", "S1_in E1_out", "W1_in S1_out"],
    ["N1_in E1_out", "E1_in N1_out", "S1_in W1_out", "W1_in S1_out"],
    ["N1_in W1_out", "E1_in S1_out", "S1_in E1_out", "W1_in N1_out"],
    ["N1_in W1_out", "N1_in E1_out", "E1_in N1_out", "W1_in S1_out"],
    ["E1_in N1_out", "E1_in S1_out", "S1_in E1_out", "N1_in W1_out"],
    ["S1_in E1_out", "N1_in W1_out", "W1_in S1_out", "E1_in N1_out"],
    ["W1_in S1_out", "W1_in N1_out", "N1_in W1_out", "S1_in E1_out"],
    ["N1_in W1_out", "N1_in S1_out", "E1_in N1_out", "S1_in E1_out"],
    ["E1_in N1_out", "E1_in W1_out", "S1_in E1_out", "W1_in S1_out"],
    ["N1_in W1_out", "S1_in E1_out", "S1_in N1_out", "W1_in S1_out"],
    ["N1_in W1_out", "E1_in N1_out", "W1_in S1_out", "W1_in E1_out"],
    ["N1_in W1_out", "N1_in S1_out", "E1_in N1_out", "S1_in E1_out", "ST_in NT_out", "NT_in ST_out"],
    ["N1_in W1_out", "S1_in E1_out", "S1_in N1_out", "W1_in S1_out", "ST_in NT_out", "NT_in ST_out"],
]

BEST_CASE_COMBINATIONS_GALERIA = [
    ["N1_in W1_out", "N1_in S1_out", "NT_in ST_out", "ST_in NT_out", "S1_in E1_out", "S1_in N1_out"],
    ["N1_in W1_out", "N1_in S1_out", "S1_in E1_out", "S1_in N1_out"],
    ["N1_in W1_out", "N1_in S1_out", "N1_in E1_out"],
    ["E1_in W1_out", "E1_in S1_out", "S1_in E1_out"],
    ["S1_in E1_out", "S1_in N1_out", "S1_in W1_out"],
    ["W1_in E1_out", "W1_in N1_out", "N1_in W1_out"],
    ["N1_in E1_out", "S1_in W1_out"]
]


# Hodoninska routes (N/S: middle+left; E/W: left and coupled right+middle; no trams)
HODONINSKA_CHOICES = [
    ["N1_in S1_out"],                 # North Middle
    ["N1_in E1_out"],                 # North Left
    ["E1_in S1_out"],                 # East Left
    ["S1_in N1_out"],                 # South Middle
    ["S1_in W1_out"],                 # South Left
    ["W1_in N1_out"],                 # West Left
    ["E1_in N1_out", "E1_in W1_out"], # East Right + East Middle
    ["W1_in S1_out", "W1_in E1_out"], # West Right + West Middle
]

HODONINSKA_NON_CONFLICTING_PAIRS = [
    (["N1_in S1_out"], ["S1_in N1_out"]),                                 # N mid + S mid
    (["E1_in N1_out", "E1_in W1_out"], ["W1_in S1_out", "W1_in E1_out"]), # E right/mid + W right/mid
    (["N1_in E1_out"], ["S1_in W1_out"]),                                 # N left + S left
    (["E1_in S1_out"], ["W1_in N1_out"]),                                 # E left + W left
]

def get_random_hodoninska_routes(is_herd=False):
    return random.choice(HODONINSKA_CHOICES)

# Aupark routes (no north-right, south-right, or trams)
AUPARK_ROUTES = [
    "N1_in S1_out", "N1_in E1_out",
    "E1_in N1_out", "E1_in W1_out", "E1_in S1_out",
    "S1_in N1_out", "S1_in W1_out",
    "W1_in S1_out", "W1_in E1_out", "W1_in N1_out",
]

AUPARK_NON_CONFLICTING_PAIRS = [
    pair for pair in NON_CONFLICTING_PAIRS
    if "N1_in W1_out" not in pair and "S1_in E1_out" not in pair and "ST_in NT_out" not in pair and "NT_in ST_out" not in pair
]

BEST_CASE_COMBINATIONS_AUPARK = [
    [r for r in combo if r not in ["N1_in W1_out", "S1_in E1_out", "ST_in NT_out", "NT_in ST_out"]]
    for combo in BEST_CASE_COMBINATIONS
]
BEST_CASE_COMBINATIONS_AUPARK = [combo for combo in BEST_CASE_COMBINATIONS_AUPARK if len(combo) > 0]


def get_random_aupark_route(is_herd=False):
    return random.choice(AUPARK_ROUTES)


def get_random_route(is_herd=False):
    # Enforce 2% tram proportion by routing 2% of NON-HERD traffic to the rails
    if not is_herd and random.random() < 0.02:
        return random.choice(RAIL_ROUTES)
    return random.choice(VEHICLE_ROUTES)

def get_vtype(is_herd=False):
    r = random.random()
    # Roughly 95% cars, 1% trucks, 4% buses (more cars in a herd)
    car_prob = 0.97 if is_herd else 0.95
    truck_prob = 0.98 if is_herd else 0.96
    
    if r < car_prob:
        return "car"
    elif r < truck_prob:
        return "truck"
    else:
        return random.choice([
            "BUS_SOR_NB12", "BUS_SOR_NB18", "BUS_SOR_NB95", 
            "BUS_SOL_U15", "BUS_SOL_U12"
        ])

# vType boilerplate: urban speeds (50 km/h cars, 40 km/h trucks/buses), SUMO-default sigma
HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<routes>

    <!-- ================================================================
         Vehicle types — urban intersection (50 km/h zone)
         ================================================================ -->
    <vType id="car"
           vClass="passenger"
           length="4.5"
           accel="2.6"
           decel="4.5"
           maxSpeed="13.89"
           minGap="2.5"
           sigma="0.5"
           speedFactor="1.0"
           speedDev="0.1"
           color="0.2,0.6,1.0"/>

    <vType id="truck"
           vClass="truck"
           length="10.0"
           accel="1.0"
           decel="3.0"
           maxSpeed="11.11"
           minGap="3.5"
           sigma="0.5"
           speedFactor="0.9"
           speedDev="0.05"
           color="0.8,0.4,0.1"/>

    <!-- Buses -->
    <vType id="BUS_SOR_NB12" vClass="bus" length="12.18" width="2.55" accel="1.067" decel="3.0" maxSpeed="11.11" minGap="3.0" sigma="0.5" speedFactor="0.95" speedDev="0.05" color="0.9,0.8,0.0"/>
    <vType id="BUS_SOR_NB18" vClass="bus" length="18.75" width="2.55" accel="1.100" decel="3.0" maxSpeed="11.11" minGap="3.0" sigma="0.5" speedFactor="0.95" speedDev="0.05" color="0.9,0.8,0.0"/>
    <vType id="BUS_SOR_NB95" vClass="bus" length="9.60" width="2.53" accel="1.067" decel="3.0" maxSpeed="11.11" minGap="3.0" sigma="0.5" speedFactor="0.95" speedDev="0.05" color="0.9,0.8,0.0"/>
    <vType id="BUS_SOL_U15" vClass="bus" length="14.89" width="2.55" accel="1.067" decel="3.0" maxSpeed="11.11" minGap="3.0" sigma="0.5" speedFactor="0.95" speedDev="0.05" color="0.9,0.8,0.0"/>
    <vType id="BUS_SOL_U12" vClass="bus" length="12.00" width="2.55" accel="1.067" decel="3.0" maxSpeed="11.11" minGap="3.0" sigma="0.5" speedFactor="0.95" speedDev="0.05" color="0.9,0.8,0.0"/>

    <!-- Trams -->
    <vType id="TRAM_VARIO" vClass="tram" length="22.60" width="2.48" accel="0.677" decel="2.0" maxSpeed="13.89" minGap="4.0" sigma="0.5" speedFactor="1.0" speedDev="0.0" color="1.0,0.2,0.2"/>

"""

FOOTER = "</routes>\n"


# --- FILE WRITER ---

def write_route_file(filename, events):
    """Sort events by depart time and write vehicle <trip>s and pedestrian <person>s as XML."""
    events.sort(key=lambda x: x[0])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(HEADER)

        f.write("    <!-- Traffic -->\n")

        veh_count = 0
        ped_count = 0

        for event in events:
            t, event_type, data = event[0], event[1], event[2]
            
            if event_type == "vehicle":
                is_herd = event[3] if len(event) > 3 else False
                
                # Check if the generated route is for a tram, assigning explicit tram vType if so
                if data in RAIL_ROUTES:
                    vtype = random.choice(["TRAM_VARIO"])
                else:
                    vtype = get_vtype(is_herd)
                
                from_edge, to_edge = data.split()
                f.write(
                    f'    <trip id="veh_{veh_count}" type="{vtype}" '
                    f'depart="{t:.1f}" from="{from_edge}" to="{to_edge}" departLane="best" departSpeed="max"/>\n'
                )
                veh_count += 1

            elif event_type == "pedestrian":
                # data is a (from_edge, to_edge) tuple
                from_edge, to_edge = data
                f.write(
                    f'    <person id="ped_{ped_count}" depart="{t:.1f}">\n'
                    f'        <walk from="{from_edge}" to="{to_edge}"/>\n'
                    f'    </person>\n'
                )
                ped_count += 1

        f.write(FOOTER)

    print(f"  {filename}: {veh_count} vehicles, {ped_count} pedestrians")


# --- GENERATION FUNCTIONS ---

def generate_calibration(filename):
    events = []

    for t in range(0, 1500, 30):
        if t % 120:
            herd_route = get_random_route(True)
            herd_size = random.randint(8, 10)
            for i in range(herd_size):
                events.append((t + i * 1.5, "vehicle", herd_route, True))
        else:
            events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES)))

    write_route_file(filename, events)

def generate_tutorial(filename):
    """Tutorial stage: single vehicles, then non-conflicting pairs, then non-conflicting herds."""
    events = []

    # Phase 1: single vehicles, 1 per 7s, no pedestrians
    for t in range(0, 800, 12):
        events.append((float(t), "vehicle", get_random_route()))

    # Phase 2: NON-CONFLICTING ROUTES ONLY, one pair per 7s
    for t in range(800, 1300, 12):
        r1, r2 = random.choice(NON_CONFLICTING_PAIRS)
        events.append((float(t), "vehicle", r1))
        events.append((float(t), "vehicle", r2))

    # Phase 3: a pair of herds on non-conflicting routes
    for herd_start in [1400.0, 1500.0]:
        hr1, hr2 = random.choice(NON_CONFLICTING_PAIRS)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            events.append((herd_start + i * 1.5, "vehicle", hr1, True))
            events.append((herd_start + i * 1.5, "vehicle", hr2, True))

    write_route_file(filename, events)


def generate_easy(filename):
    events = []
    
    for t in range(0, 1500, 15):
        herd_routes = random.choice(BEST_CASE_COMBINATIONS)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            for r in herd_routes:
                events.append((float(t) + i * 1.5, "vehicle", r, True))

    write_route_file(filename, events)


def generate_light(filename):
    """Stage 2: 2-3 vehicles (any direction) every 4s plus 1 pedestrian every 20s."""
    events = []

    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_route()))

    for t in range(0, 1501, 20):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES)))

    write_route_file(filename, events)


def generate_heavy(filename):
    """Stage 3: light flow plus herds of 8-12 cars every 120s and 1 pedestrian every 15s."""
    events = []

    # Base heavy flow (same pattern as light)
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_route()))

    # Slightly denser pedestrians than light
    for t in range(0, 1501, 15):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES)))

    # Herds every 2 minutes: t=120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440
    for herd_start in range(120, 1501, 120):
        herd_route = get_random_route(True)
        herd_size  = random.randint(8, 12)
        for i in range(herd_size):
            events.append((float(herd_start) + i * 1.2, "vehicle", herd_route, True))

    write_route_file(filename, events)


# --- HODONINSKA GENERATION FUNCTIONS ---

def generate_hodoninska_tutorial(filename):
    events = []
    # Phase 1: single sets of vehicles, 1 per 7s, no pedestrians
    for t in range(0, 800, 8):
        routes = get_random_hodoninska_routes()
        for r in routes:
            events.append((float(t), "vehicle", r))

    # Phase 2: NON-CONFLICTING ROUTES ONLY, one pair per 7s
    for t in range(800, 1300, 8):
        group1, group2 = random.choice(HODONINSKA_NON_CONFLICTING_PAIRS)
        for r in group1 + group2:
            events.append((float(t), "vehicle", r))

    # Phase 3: a pair of herds on non-conflicting routes
    for herd_start in [1400.0, 1500.0]:
        group1, group2 = random.choice(HODONINSKA_NON_CONFLICTING_PAIRS)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            for r in group1 + group2:
                events.append((herd_start + i * 1.5, "vehicle", r, True))

    write_route_file(filename, events)

def generate_hodoninska_easy(filename):
    events = []
    
    for t in range(0, 1500, 30):
        group1, group2 = random.choice(HODONINSKA_NON_CONFLICTING_PAIRS)
        herd_routes = group1 + group2
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            for r in herd_routes:
                events.append((float(t) + i * 1.5, "vehicle", r, True))

    write_route_file(filename, events)

def generate_hodoninska_light(filename):
    events = []
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            routes = get_random_hodoninska_routes()
            for r in routes:
                events.append((float(t), "vehicle", r))
    write_route_file(filename, events)

def generate_hodoninska_heavy(filename):
    events = []
    # Base heavy flow
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            routes = get_random_hodoninska_routes()
            for r in routes:
                events.append((float(t), "vehicle", r))

    # Herds every 2 minutes
    for herd_start in range(120, 1501, 120):
        herd_routes = get_random_hodoninska_routes(True)
        herd_size = random.randint(8, 12)
        for i in range(herd_size):
            for r in herd_routes:
                events.append((float(herd_start) + i * 1.2, "vehicle", r, True))

    write_route_file(filename, events)


# --- AUPARK GENERATION FUNCTIONS ---

def generate_aupark_tutorial(filename):
    events = []
    for t in range(0, 800, 8):
        events.append((float(t), "vehicle", get_random_aupark_route()))

    for t in range(800, 1300, 8):
        r1, r2 = random.choice(AUPARK_NON_CONFLICTING_PAIRS)
        events.append((float(t), "vehicle", r1))
        events.append((float(t), "vehicle", r2))

    for herd_start in [1400.0, 1500.0]:
        hr1, hr2 = random.choice(AUPARK_NON_CONFLICTING_PAIRS)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            events.append((herd_start + i * 1.5, "vehicle", hr1, True))
            events.append((herd_start + i * 1.5, "vehicle", hr2, True))

    write_route_file(filename, events)

def generate_aupark_easy(filename):
    events = []
    
    for t in range(0, 1500, 30):
        herd_routes = random.choice(BEST_CASE_COMBINATIONS_AUPARK)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            for r in herd_routes:
                events.append((float(t) + i * 1.5, "vehicle", r, True))

    write_route_file(filename, events)

def generate_aupark_light(filename):
    events = []
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_aupark_route()))

    for t in range(0, 1501, 20):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES_CANONICAL)))

    write_route_file(filename, events)

def generate_aupark_heavy(filename):
    events = []
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_aupark_route()))

    for t in range(0, 1501, 15):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES_CANONICAL)))

    for herd_start in range(120, 1501, 120):
        herd_route = get_random_aupark_route(True)
        herd_size  = random.randint(8, 12)
        for i in range(herd_size):
            events.append((float(herd_start) + i * 1.2, "vehicle", herd_route, True))

    write_route_file(filename, events)

# Galeria routes (no east-right or west-right)
GALERIA_ROUTES = [
    "N1_in W1_out", "N1_in S1_out", "N1_in E1_out",
    "E1_in W1_out", "E1_in S1_out",
    "S1_in E1_out", "S1_in N1_out", "S1_in W1_out",
    "W1_in E1_out", "W1_in N1_out",
]

GALERIA_NON_CONFLICTING_PAIRS = [
    pair for pair in NON_CONFLICTING_PAIRS
    if "E1_in N1_out" not in pair and "W1_in S1_out" not in pair
]

def get_random_galeria_route(is_herd=False):
    if not is_herd and random.random() < 0.02:
        return random.choice(RAIL_ROUTES)
    return random.choice(GALERIA_ROUTES)

def generate_galeria_tutorial(filename):
    events = []
    for t in range(0, 800, 12):
        events.append((float(t), "vehicle", get_random_galeria_route()))

    for t in range(800, 1300, 12):
        r1, r2 = random.choice(GALERIA_NON_CONFLICTING_PAIRS)
        events.append((float(t), "vehicle", r1))
        events.append((float(t), "vehicle", r2))

    for herd_start in [1400.0, 1500.0]:
        hr1, hr2 = random.choice(GALERIA_NON_CONFLICTING_PAIRS)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            events.append((herd_start + i * 1.5, "vehicle", hr1, True))
            events.append((herd_start + i * 1.5, "vehicle", hr2, True))

    write_route_file(filename, events)

def generate_galeria_easy(filename):
    events = []
    for t in range(0, 1500, 32):
        herd_routes = random.choice(BEST_CASE_COMBINATIONS_GALERIA)
        if "NT_in ST_out" in herd_routes or "ST_in NT_out" in herd_routes:
            if random.random() < 0.4:
                herd_routes = random.choice(BEST_CASE_COMBINATIONS_GALERIA)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            for r in herd_routes:
                events.append((float(t) + i * 1.5, "vehicle", r, True))

    write_route_file(filename, events)

def generate_galeria_light(filename):
    events = []
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_galeria_route()))

    for t in range(0, 1501, 20):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES_CANONICAL)))

    write_route_file(filename, events)

def generate_galeria_heavy(filename):
    events = []
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_galeria_route()))

    for t in range(0, 1501, 15):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES_CANONICAL)))

    for herd_start in range(120, 1501, 120):
        herd_route = get_random_galeria_route(True)
        herd_size = random.randint(8, 12)
        for i in range(herd_size):
            events.append((float(herd_start) + i * 1.2, "vehicle", herd_route, True))

    write_route_file(filename, events)

# --- CANONICAL GENERATION FUNCTIONS ---

def generate_canonical_tutorial(filename):
    events = []

    # Phase 1: single vehicles, 1 per 7s, no pedestrians
    for t in range(0, 800, 12):
        events.append((float(t), "vehicle", get_random_route()))

    # Phase 2: NON-CONFLICTING ROUTES ONLY, one pair per 7s
    for t in range(800, 1300, 12):
        r1, r2 = random.choice(NON_CONFLICTING_PAIRS)
        events.append((float(t), "vehicle", r1))
        events.append((float(t), "vehicle", r2))

    # Phase 3: a pair of herds on non-conflicting routes
    for herd_start in [1400.0, 1500.0]:
        hr1, hr2 = random.choice(NON_CONFLICTING_PAIRS)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            events.append((herd_start + i * 1.5, "vehicle", hr1, True))
            events.append((herd_start + i * 1.5, "vehicle", hr2, True))

    write_route_file(filename, events)


def generate_canonical_easy(filename):
    events = []
    
    for t in range(0, 1500, 30):
        herd_routes = random.choice(BEST_CASE_COMBINATIONS)
        herd_size = random.randint(8, 10)
        for i in range(herd_size):
            for r in herd_routes:
                events.append((float(t) + i * 1.5, "vehicle", r, True))

    write_route_file(filename, events)


def generate_canonical_light(filename):
    events = []

    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_route()))

    # 1 pedestrian every 20 seconds -> canonical routes
    for t in range(0, 1501, 20):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES_CANONICAL)))

    write_route_file(filename, events)


def generate_canonical_heavy(filename):
    events = []

    # Base heavy flow
    for t in range(0, 1501, 4):
        n = random.randint(2, 3)
        for _ in range(n):
            events.append((float(t), "vehicle", get_random_route()))

    # Slightly denser pedestrians than light -> canonical routes
    for t in range(0, 1501, 15):
        events.append((float(t), "pedestrian", random.choice(PEDESTRIAN_ROUTES_CANONICAL)))

    # Herds every 2 minutes
    for herd_start in range(120, 1501, 120):
        herd_route = get_random_route(True)
        herd_size  = random.randint(8, 12)
        for i in range(herd_size):
            events.append((float(herd_start) + i * 1.2, "vehicle", herd_route, True))

    write_route_file(filename, events)

# --- ENTRY POINT ---

if __name__ == "__main__":
    print("Generating SUMO curriculum route files...")
    
    # Change this prefix to toggle between intersection files without overwriting
    ROUTE_PREFIX = "Hornbach"
    
    #generate_tutorial(f"simulation/{ROUTE_PREFIX}-tutorial.rou.xml")
    # generate_easy(f"simulation/{ROUTE_PREFIX}-easy.rou.xml")
    # generate_light(f"simulation/{ROUTE_PREFIX}-medium.rou.xml")
    # generate_heavy(f"simulation/{ROUTE_PREFIX}-hard.rou.xml")
    # generate_tutorial(f"simulation/{ROUTE_PREFIX}-tutorial.rou.xml")
    # generate_calibration(f"simulation/{ROUTE_PREFIX}-calibration.rou.xml")

    # Example for generating Hodoninska routes:
    HODONINSKA_PREFIX = "Hodoninska"
    # generate_hodoninska_tutorial(f"simulation/{HODONINSKA_PREFIX}-tutorial.rou.xml")
    # generate_hodoninska_easy(f"simulation/{HODONINSKA_PREFIX}-easy.rou.xml")
    # generate_hodoninska_light(f"simulation/{HODONINSKA_PREFIX}-medium.rou.xml")
    # generate_hodoninska_heavy(f"simulation/{HODONINSKA_PREFIX}-hard.rou.xml")

    # Example for generating Aupark routes:
    AUPARK_PREFIX = "Aupark"
    generate_aupark_tutorial(f"simulation/{AUPARK_PREFIX}-tutorial.rou.xml")
    generate_aupark_easy(f"simulation/{AUPARK_PREFIX}-easy.rou.xml")
    generate_aupark_light(f"simulation/{AUPARK_PREFIX}-medium.rou.xml")
    generate_aupark_heavy(f"simulation/{AUPARK_PREFIX}-hard.rou.xml")

    # Example for generating Galeria routes:
    GALERIA_PREFIX = "Galeria"
    # generate_galeria_tutorial(f"simulation/{GALERIA_PREFIX}-tutorial.rou.xml")
    # generate_galeria_easy(f"simulation/{GALERIA_PREFIX}-easy.rou.xml")
    # generate_galeria_light(f"simulation/{GALERIA_PREFIX}-medium.rou.xml")
    # generate_galeria_heavy(f"simulation/{GALERIA_PREFIX}-hard.rou.xml")

    # Example for generating Canonical routes:
    CANONICAL_PREFIX = "Canonical"
    # generate_canonical_tutorial(f"simulation/{CANONICAL_PREFIX}-tutorial.rou.xml")
    # generate_canonical_easy(f"simulation/{CANONICAL_PREFIX}-easy.rou.xml")
    # generate_canonical_light(f"simulation/{CANONICAL_PREFIX}-medium.rou.xml")
    # generate_canonical_heavy(f"simulation/{CANONICAL_PREFIX}-hard.rou.xml")

    print(f"\nDone. Files generated with prefix '{ROUTE_PREFIX}'.")
