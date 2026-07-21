import pandas as pd
import random
import os

VTYPE_MAPPING = {
    'C': 'car', 'B': 'bus', 'A': 'articulated_bus', 'H': 'truck', 'T': 'tram', 'P': 'pedestrian'
}

from generate_routes import HEADER, FOOTER

def generate_density_routes(csv_path: str, out_path: str, row_index: int, scale: float = 1.0, duration: int = 1800):
    df = pd.read_csv(csv_path, header=None)
    
    categories = df.iloc[0].fillna('')
    approaches = df.iloc[1].fillna('')
    lanes = df.iloc[2].fillna('')
    vtypes = df.iloc[3].fillna('')
    
    data_row = df.iloc[row_index]
    
    events = []
    
    veh_counter = 0
    ped_counter = 0
    
    for col_idx in range(2, len(data_row)):
        val = data_row[col_idx]
        if pd.notna(val) and str(val).strip().isdigit():
            count_per_hour = int(val)
            if count_per_hour <= 0:
                continue
                
            # total count for the duration, scaled
            total_count = int(count_per_hour * scale * (duration / 3600.0))
            
            cat = categories[col_idx].strip()
            approach = approaches[col_idx].strip()
            lane = lanes[col_idx].strip()
            vtype_code = vtypes[col_idx].strip()
            
            sumo_vtype_base = VTYPE_MAPPING.get(vtype_code, 'car')
            
            if cat == 'V':
                # Map to the specific vTypes defined in HEADER
                if sumo_vtype_base == 'car':
                    sumo_vtype = 'car'
                elif sumo_vtype_base == 'truck':
                    sumo_vtype = 'truck'
                elif sumo_vtype_base == 'articulated_bus':
                    sumo_vtype = 'BUS_SOR_NB18'
                elif sumo_vtype_base == 'bus':
                    sumo_vtype = random.choice(['BUS_SOR_NB12', 'BUS_SOR_NB95', 'BUS_SOL_U15', 'BUS_SOL_U12'])
                elif sumo_vtype_base == 'tram':
                    sumo_vtype = 'TRAM_VARIO'
                else:
                    sumo_vtype = 'car'
                
                # Approximate exit mapping (just for scaffolding)
                turn_map = {
                    'N': {'R': 'W', 'M': 'S', 'L': 'E', 'T': 'S'},
                    'E': {'R': 'N', 'M': 'W', 'L': 'S', 'T': 'W'},
                    'S': {'R': 'E', 'M': 'N', 'L': 'W', 'T': 'N'},
                    'W': {'R': 'S', 'M': 'E', 'L': 'N', 'T': 'E'}
                }
                exit_dir = turn_map.get(approach, {}).get(lane, 'Unknown')
                entry_edge = f"{approach}1_in" if sumo_vtype_base != 'tram' else f"{approach}T_in"
                exit_edge = f"{exit_dir}1_out" if sumo_vtype_base != 'tram' else f"{approach}T_out"
                
                for _ in range(total_count):
                    spawn_time = random.uniform(0, duration)
                    events.append((spawn_time, "vehicle", f"{entry_edge} {exit_edge}", sumo_vtype))
                    veh_counter += 1
            elif cat == 'P':
                # --- PEDESTRIAN LOGIC ---
                if approach == "N":
                    if lane in ['ML', 'T', 'D']:
                        ped_start_edge = f"PNEI_in"
                        ped_end_edge = f"PNWI_out"
                    elif lane == "R":
                        ped_start_edge = f"PNW_in"
                        ped_end_edge = f"PNWI_in"
                    else:
                        print(f"Unrecognized lane {lane} for approach {approach}")
                        ped_start_edge = f"ped_start_{approach}_{lane}"
                        ped_end_edge = f"ped_end_{approach}_{lane}"
                elif approach == "E":
                    if lane in ['ML', 'T', 'D']:
                        ped_start_edge = f"PSEI_in"
                        ped_end_edge = f"PNEI_out"
                    elif lane == "R":
                        ped_start_edge = f"PNE_in"
                        ped_end_edge = f"PNEI_in"
                    else:
                        print(f"Unrecognized lane {lane} for approach {approach}")
                        ped_start_edge = f"ped_start_{approach}_{lane}"
                        ped_end_edge = f"ped_end_{approach}_{lane}"
                elif approach == "S":
                    if lane in ['ML', 'T', 'D']:
                        ped_start_edge = f"PSWI_in"
                        ped_end_edge = f"PSEI_out"
                    elif lane == "R":
                        ped_start_edge = f"PSE_in"
                        ped_end_edge = f"PSEI_in"
                    else:
                        print(f"Unrecognized lane {lane} for approach {approach}")
                        ped_start_edge = f"ped_start_{approach}_{lane}"
                        ped_end_edge = f"ped_end_{approach}_{lane}"
                elif approach == "W":
                    if lane in ['ML', 'T', 'D']:
                        ped_start_edge = f"PNWI_in"
                        ped_end_edge = f"PSWI_out"
                    elif lane == "R":
                        ped_start_edge = f"PSW_in"
                        ped_end_edge = f"PSWI_in"
                    else:
                        print(f"Unrecognized lane {lane} for approach {approach}")
                        ped_start_edge = f"ped_start_{approach}_{lane}"
                        ped_end_edge = f"ped_end_{approach}_{lane}"
                else:
                    print(f"Unrecognized lane {lane} for approach {approach}")
                    ped_start_edge = f"ped_start_{approach}_{lane}"
                    ped_end_edge = f"ped_end_{approach}_{lane}"

                for _ in range(total_count):
                    spawn_time = random.uniform(0, duration)
                    events.append((spawn_time, "pedestrian", (ped_start_edge, ped_end_edge)))
                    ped_counter += 1
                    
    events.sort(key=lambda x: x[0])
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HEADER)
        f.write("    <!-- Traffic -->\n")
        
        v_idx, p_idx = 0, 0
        for event in events:
            t, event_type = event[0], event[1]
            if event_type == "vehicle":
                route_str = event[2]
                from_edge, to_edge = route_str.split()
                f.write(f'    <trip id="veh_{v_idx}" type="{event[3]}" depart="{t:.1f}" from="{from_edge}" to="{to_edge}" departLane="best" departSpeed="max"/>\n')
                v_idx += 1
            else:
                f.write(f'    <person id="ped_{p_idx}" depart="{t:.1f}">\n        <walk from="{event[2][0]}" to="{event[2][1]}"/>\n    </person>\n')
                p_idx += 1
        
        f.write(FOOTER)
    
    return data_row[1] # Return the time string name for this scenario (e.g. 20:00)

generate_density_routes("simulation/TrafficDensity/Aupark.csv", "simulation/TrafficDensity/Ap-test.rou.xml", 4, 0.2)