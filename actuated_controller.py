"""Vehicle-actuated (gap-out / max-out) baseline over the shared canonical phase library."""

from __future__ import annotations

import traci
from TrafficLightController import VEHICLE_SLOT_COUNT


class ActuatedController:
    def __init__(
        self,
        controller,
        route_slots,
        cycle_phases,
        min_green: float = 5.0,
        max_green: float = 45.0,
        max_gap: float = 3.0,
        step_length: float = 1.0,
    ):
        """Candidate phases, min/max green, gap-out threshold and step length (all seconds)."""
        self.controller = controller
        self.route_slots = route_slots
        self.cycle = [list(p) for p in cycle_phases]
        if not self.cycle:
            raise ValueError("ActuatedController requires a non-empty cycle_phases list.")

        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.max_gap = float(max_gap)
        self.step_length = float(step_length)

        self.phase_lanes: list[list[str]] = []
        self._resolved = False
        self.idx = 0
        self.green_time = 0.0
        self.gap_time = 0.0

    def reset(self) -> None:
        """Reset internal timers and force lane re-resolution for a new run."""
        self.idx = 0
        self.green_time = 0.0
        self.gap_time = 0.0
        self._resolved = False
        self.phase_lanes = []

    def _resolve_lanes(self) -> None:
        """Map each phase's green vehicular slots to existing SUMO lane ids."""
        try:
            existing = set(traci.lane.getIDList())
        except Exception:
            existing = set()

        self.phase_lanes = []
        for phase in self.cycle:
            lanes: set[str] = set()
            for slot_i in range(VEHICLE_SLOT_COUNT):
                if slot_i < len(phase) and phase[slot_i] == 1:
                    rs = self.route_slots[slot_i]
                    for fe in getattr(rs, "feed_edges", []) or []:
                        edge_id = getattr(fe, "edge_id", "")
                        lane_indices = getattr(fe, "lane_indices", []) or []
                        for li in lane_indices:
                            cand = f"{edge_id}_{li}"
                            if not existing or cand in existing:
                                lanes.add(cand)
                            elif edge_id in existing:
                                lanes.add(edge_id)
            self.phase_lanes.append(sorted(lanes))
        self._resolved = True

    def _demand_on(self, lanes: list[str]) -> int:
        """Number of vehicles currently on the given lanes."""
        n = 0
        for ln in lanes:
            try:
                n += traci.lane.getLastStepVehicleNumber(ln)
            except Exception:
                continue
        return n

    def _next_phase_with_demand(self) -> int:
        """Next phase in the ring that has waiting demand; fall back to next."""
        n = len(self.cycle)
        for off in range(1, n + 1):
            j = (self.idx + off) % n
            if self._demand_on(self.phase_lanes[j]) > 0:
                return j
        return (self.idx + 1) % n

    def decide(self) -> list[int]:
        """Return the canonical phase array to apply at this step."""
        if not self._resolved:
            self._resolve_lanes()

        demand = self._demand_on(self.phase_lanes[self.idx])

        self.green_time += self.step_length
        if demand > 0:
            self.gap_time = 0.0
        else:
            self.gap_time += self.step_length

        if self.green_time >= self.min_green:
            max_out = self.green_time >= self.max_green
            gap_out = self.gap_time >= self.max_gap
            if max_out or gap_out:
                self.idx = self._next_phase_with_demand()
                self.green_time = 0.0
                self.gap_time = 0.0

        return self.cycle[self.idx]
