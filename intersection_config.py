import os
import yaml
from pathlib import Path
from TrafficLightController import (
    TrafficLightController,
    LaneSignalMapping,
    PhaseDefinition,
    TransitionTiming,
    SlotType,
    VEHICLE_SLOT_NAMES,
    PEDESTRIAN_SLOT_NAMES,
    CANONICAL_SLOT_COUNT
)
from IntersectionRoutes import RouteSlot, FeedEdge

def load_intersection_config(yaml_path: str, tracker=None) -> tuple[TrafficLightController, list[RouteSlot], dict]:
    """Load a YAML intersection config; return (controller, ordered route_slots, normalization bounds)."""
    # Accept a bare filename, falling back to the configs/ folder
    if not os.path.exists(yaml_path):
        alt = os.path.join("configs", yaml_path)
        if os.path.exists(alt):
            yaml_path = alt
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    inter_config = config.get("intersection", {})
    sumo_id = inter_config.get("sumo_id", "0")
    sumocfg_path = inter_config.get("sumocfg_path", "")
    signal_count = inter_config.get("sumo_signal_count", 0)
    verbose = inter_config.get("verbose", False)
    verbose_selection = inter_config.get("verbose_phase_selection", False)

    bounds = inter_config.get("normalization_bounds", {
        "max_vehicles_in_lane": 20,
        "max_wait_per_vehicle": 90.0,
        "max_light_age": 60.0
    })
    
    bounds["sumocfg_path"] = sumocfg_path  # keep the sumocfg path accessible via bounds

    timing_config = inter_config.get("timing", {})
    timing = TransitionTiming(
        yellow_steps=timing_config.get("yellow_steps", 3),
        all_red_steps=timing_config.get("all_red_steps", 1),
        red_yellow_steps=timing_config.get("red_yellow_steps", 2),
        crosswalk_extra_red_steps=timing_config.get("crosswalk_extra_red_steps", 0),
        green_steps=timing_config.get("green_steps", 0),
    )

    lane_map = []
    ordered_route_slots = []

    def process_slots(slots_data: dict, slot_type: SlotType, expected_names: list[str]):
        # Iterate over expected canonical combinations
        for name in expected_names:
            slot_data = slots_data.get(name, {})
            
            sumo_tls_indices = slot_data.get("sumo_tls_indices", [])
            feed_edges_data = slot_data.get("feed_edges", [])
            feed_edges = []
            if feed_edges_data:
                for edge_data in feed_edges_data:
                    feed_edges.append(FeedEdge(
                        edge_id=edge_data.get("edge_id", ""),
                        lane_indices=edge_data.get("lane_indices", [])
                    ))

            approaches_data = slot_data.get("approaches", [])
            approaches = []
            if approaches_data:
                for app_data in approaches_data:
                    approach = []
                    for edge_data in app_data:
                        approach.append(FeedEdge(
                            edge_id=edge_data.get("edge_id", ""),
                            lane_indices=edge_data.get("lane_indices", [])
                        ))
                    approaches.append(approach)
            
            active = len(sumo_tls_indices) > 0
            max_capacity = slot_data.get("max_capacity", 0)
            
            lane_map.append(
                LaneSignalMapping(
                    name=name,
                    slot_type=slot_type,
                    sumo_tls_indices=sumo_tls_indices,
                    sumo_lane_ids=[], # Minimalist mapping uses RouteSlot fallback
                    active=active,
                    max_capacity=max_capacity
                )
            )
            
            route_slot = RouteSlot(name=name, feed_edges=feed_edges, approaches=approaches)
            ordered_route_slots.append(route_slot)

    process_slots(config.get("vehicular_slots", {}), SlotType.VEHICLE, VEHICLE_SLOT_NAMES)
    process_slots(config.get("pedestrian_slots", {}), SlotType.PEDESTRIAN, PEDESTRIAN_SLOT_NAMES)

    # Valid phases
    valid_phases = []
    for phase_data in config.get("valid_phases", []):
        label = phase_data.get("label", "Unknown")
        signals = phase_data.get("signals", [])
        if len(signals) != CANONICAL_SLOT_COUNT:
            raise ValueError(f"Phase '{label}' does not have {CANONICAL_SLOT_COUNT} binary signals.")
        valid_phases.append(PhaseDefinition(signals, label))

    # Standard cycle
    standard_cycle = []
    for phase_data in config.get("standard_cycle", []):
        label = phase_data.get("label", "Unknown")
        signals = phase_data.get("signals", [])
        if len(signals) != CANONICAL_SLOT_COUNT:
            raise ValueError(f"Phase '{label}' in standard_cycle does not have {CANONICAL_SLOT_COUNT} binary signals.")
        standard_cycle.append(PhaseDefinition(signals, label))

    from TrafficLightController import PhaseTracker
    tracker = PhaseTracker(valid_phases)

    controller = TrafficLightController(
        sumo_id=sumo_id,
        lane_map=lane_map,
        valid_phases=valid_phases,
        sumo_signal_count=signal_count,
        timing=timing,
        tracker=tracker,
        verbose=verbose,
        verbose_phase_selection=verbose_selection,
        standard_cycle=standard_cycle if standard_cycle else None,
    )

    return controller, ordered_route_slots, bounds
