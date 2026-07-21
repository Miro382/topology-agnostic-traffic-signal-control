"""Canonical route slots: the full set of feeder lanes a vehicle occupies approaching the intersection."""

from __future__ import annotations
from dataclasses import dataclass, field
import traci


@dataclass
class FeedEdge:
    """One edge in a feeder chain plus the 0-based lane indices that carry this movement."""
    edge_id:      str
    lane_indices: list[int]

    def lane_ids(self) -> list[str]:
        return [f"{self.edge_id}_{i}" for i in self.lane_indices]


@dataclass
class RouteSlot:
    """Feeder chain for one canonical (direction, movement) pair; use `approaches` for multi-approach slots."""
    name:        str
    feed_edges:  list[FeedEdge] = field(default_factory=list)
    approaches:  list[list[FeedEdge]] = field(default_factory=list)

    def __post_init__(self):
        # Normalise feed_edges to a single approach
        if not self.approaches and self.feed_edges:
            self.approaches = [self.feed_edges]

    def all_lane_ids(self) -> list[str]:
        """All SUMO lane ID strings across every physical approach."""
        ids = []
        for approach in self.approaches:
            for fe in approach:
                ids.extend(fe.lane_ids())
        return ids

    def distance_to_stop_line(self, lane_id: str, vehicle_position: float) -> float:
        """Remaining distance from the vehicle to the stop line along its feeder approach chain."""
        edge_of_lane = "_".join(lane_id.split("_")[:-1])   # strip trailing _N

        for approach in self.approaches:
            if any(fe.edge_id == edge_of_lane for fe in approach):
                remaining = 0.0
                past_current = False
                for fe in approach:
                    if fe.edge_id == edge_of_lane:
                        lane_length   = traci.lane.getLength(lane_id)
                        remaining    += lane_length - vehicle_position
                        past_current  = True
                    elif past_current:
                        representative_lane = f"{fe.edge_id}_{fe.lane_indices[0]}"
                        remaining += traci.lane.getLength(representative_lane)
                return remaining

        return 0.0

    def total_length(self) -> float:
        """Normalisation length: the longest of the slot's independent approaches."""
        max_len = 0.0
        for approach in self.approaches:
            approach_len = 0.0
            for fe in approach:
                representative = f"{fe.edge_id}_{fe.lane_indices[0]}"
                approach_len += traci.lane.getLength(representative)
            max_len = max(max_len, approach_len)
        return max_len


# Aupark route slots, derived from the net.xml connections (lane indices are 0-based)
def build_aupark_route_slots() -> dict[str, RouteSlot]:
    """RouteSlots for the Aupark intersection, keyed by canonical slot label."""
    slots: dict[str, RouteSlot] = {}

    # EAST arm (E3_in_0 is the pedestrian lane, not a vehicle lane)
    slots["East/Straight"] = RouteSlot(
        name       = "East/Straight",
        feed_edges = [
            FeedEdge("E1_in", lane_indices=[1, 2]),   # E1_in has 3 lanes (0,1,2); str lanes are 1,2
            FeedEdge("E2_in", lane_indices=[1, 2, 3, 4, 5]),  # most lanes feed E3_in straight
            FeedEdge("E3_in", lane_indices=[1, 2]),   # straight movement
        ],
    )
    slots["East/Left"] = RouteSlot(
        name       = "East/Left",
        feed_edges = [
            FeedEdge("E1_in", lane_indices=[1, 2]),
            FeedEdge("E2_in", lane_indices=[1, 2, 3, 4, 5]),
            FeedEdge("E3_in", lane_indices=[3, 4]),   # left-turn movement
        ],
    )
    slots["East/Right"] = RouteSlot(
        name       = "East/Right",
        feed_edges = [
            FeedEdge("E1_in", lane_indices=[1, 2]),
            FeedEdge("E2_in", lane_indices=[1]),      # right-turn peels off at J29
            FeedEdge("NE_in", lane_indices=[0, 1]),   # sub-junction approach
        ],
    )

    # SOUTH arm
    slots["South/Straight"] = RouteSlot(
        name       = "South/Straight",
        feed_edges = [
            FeedEdge("S1_in", lane_indices=[1, 2, 3]),
            FeedEdge("S2_in", lane_indices=[1, 2, 3, 4, 5]),
            FeedEdge("S3_in", lane_indices=[1, 2]),
        ],
    )
    slots["South/Left"] = RouteSlot(
        name       = "South/Left",
        feed_edges = [
            FeedEdge("S1_in", lane_indices=[1, 2, 3]),
            FeedEdge("S2_in", lane_indices=[1, 2, 3, 4, 5]),
            FeedEdge("S3_in", lane_indices=[3, 4]),
        ],
    )
    slots["South/Right"] = RouteSlot(
        name       = "South/Right",
        feed_edges = [
            FeedEdge("S1_in", lane_indices=[1, 2, 3]),
            FeedEdge("S2_in", lane_indices=[1]),
            FeedEdge("SE_in", lane_indices=[0, 1]),
        ],
    )

    # WEST arm
    slots["West/Straight"] = RouteSlot(
        name       = "West/Straight",
        feed_edges = [
            FeedEdge("W1_in", lane_indices=[1, 2, 3, 4, 5]),
            FeedEdge("W2_in", lane_indices=[1, 2]),
        ],
    )
    slots["West/Left"] = RouteSlot(
        name       = "West/Left",
        feed_edges = [
            FeedEdge("W1_in", lane_indices=[1, 2, 3, 4, 5]),
            FeedEdge("W2_in", lane_indices=[3, 4]),
        ],
    )
    slots["West/Right"] = RouteSlot(
        name       = "West/Right",
        feed_edges = [
            FeedEdge("W1_in", lane_indices=[1]),
            FeedEdge("SW_in", lane_indices=[0, 1]),
        ],
    )

    # NORTH arm
    slots["North/Straight"] = RouteSlot(
        name       = "North/Straight",
        feed_edges = [
            FeedEdge("N1_in", lane_indices=[1, 2, 3, 4, 5]),
            FeedEdge("N2_in", lane_indices=[1, 2]),
        ],
    )
    slots["North/Left"] = RouteSlot(
        name       = "North/Left",
        feed_edges = [
            FeedEdge("N1_in", lane_indices=[1, 2, 3, 4, 5]),
            FeedEdge("N2_in", lane_indices=[3, 4]),
        ],
    )
    slots["North/Right"] = RouteSlot(
        name       = "North/Right",
        feed_edges = [
            FeedEdge("N1_in", lane_indices=[1]),
            FeedEdge("NW_in", lane_indices=[0, 1]),
        ],
    )

    # PEDESTRIAN slots
    slots["North/Ped/Right"]  = RouteSlot("North/Ped/Right",  [FeedEdge(":M_w0",  [0])])
    slots["North/Ped/Middle"] = RouteSlot("North/Ped/Middle", [FeedEdge(":M_w1",  [0])])
    slots["North/Ped/Left"]   = RouteSlot("North/Ped/Left",   [FeedEdge(":M_w2",  [0])])

    slots["East/Ped/Right"]   = RouteSlot("East/Ped/Right",   [FeedEdge(":M_w3",  [0])])
    slots["East/Ped/Middle"]  = RouteSlot("East/Ped/Middle",  [FeedEdge(":M_w4",  [0])])
    slots["East/Ped/Left"]    = RouteSlot("East/Ped/Left",    [FeedEdge(":M_w5",  [0])])

    slots["South/Ped/Right"]  = RouteSlot("South/Ped/Right",  [FeedEdge(":M_w6",  [0])])
    slots["South/Ped/Middle"] = RouteSlot("South/Ped/Middle", [FeedEdge(":M_w7",  [0])])
    slots["South/Ped/Left"]   = RouteSlot("South/Ped/Left",   [FeedEdge(":M_w8",  [0])])

    slots["West/Ped/Right"]   = RouteSlot("West/Ped/Right",   [FeedEdge(":M_w9",  [0])])
    slots["West/Ped/Middle"]  = RouteSlot("West/Ped/Middle",  [FeedEdge(":M_w10", [0])])
    slots["West/Ped/Left"]    = RouteSlot("West/Ped/Left",    [FeedEdge(":M_w11", [0])])

    return slots