"""Training-free max-pressure baseline (Varaiya, 2013) over the shared canonical phase library."""

from __future__ import annotations

import traci
from TrafficLightController import VEHICLE_SLOT_COUNT


class MaxPressureController:
    def __init__(
        self,
        controller,
        route_slots,
        cycle_phases,
        min_green: float = 5.0,
        max_green: float = 45.0,
        step_length: float = 1.0,
    ):
        """Candidate phases plus min/max green (s) and step length; route_slots gives slot -> feed-lane mapping."""
        self.controller = controller
        self.route_slots = route_slots
        self.cycle = [list(p) for p in cycle_phases]
        if not self.cycle:
            raise ValueError("MaxPressureController requires a non-empty cycle_phases list.")

        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.step_length = float(step_length)

        # Per phase: incoming lanes served and the downstream lanes they feed
        self.phase_in: list[list[str]] = []
        self.phase_out: list[list[str]] = []
        self._resolved = False
        self.idx = 0
        self.green_time = 0.0

    def reset(self) -> None:
        """Reset internal state and force lane re-resolution for a new run."""
        self.idx = 0
        self.green_time = 0.0
        self._resolved = False
        self.phase_in = []
        self.phase_out = []

    def _resolve_lanes(self) -> None:
        """Map each phase's green vehicular slots to incoming and downstream lanes."""
        try:
            existing = set(traci.lane.getIDList())
        except Exception:
            existing = set()

        self.phase_in = []
        self.phase_out = []
        for phase in self.cycle:
            incoming: set[str] = set()
            for slot_i in range(VEHICLE_SLOT_COUNT):
                if slot_i < len(phase) and phase[slot_i] == 1:
                    rs = self.route_slots[slot_i]
                    for fe in getattr(rs, "feed_edges", []) or []:
                        edge_id = getattr(fe, "edge_id", "")
                        lane_indices = getattr(fe, "lane_indices", []) or []
                        for li in lane_indices:
                            cand = f"{edge_id}_{li}"
                            if not existing or cand in existing:
                                incoming.add(cand)
                            elif edge_id in existing:
                                incoming.add(edge_id)
            incoming_list = sorted(incoming)

            # Resolve the downstream lane(s) each incoming lane discharges to.
            downstream: set[str] = set()
            for ln in incoming_list:
                try:
                    for link in traci.lane.getLinks(ln):
                        # link[0] is the destination (downstream) lane id
                        if link and link[0]:
                            downstream.add(link[0])
                except Exception:
                    continue

            self.phase_in.append(incoming_list)
            self.phase_out.append(sorted(downstream))
        self._resolved = True

    @staticmethod
    def _halting(lanes: list[str]) -> int:
        """Total halting (queued) vehicles on the given lanes."""
        n = 0
        for ln in lanes:
            try:
                n += traci.lane.getLastStepHaltingNumber(ln)
            except Exception:
                continue
        return n

    def _pressure(self, i: int) -> int:
        """Pressure of phase i: incoming queue minus downstream queue."""
        return self._halting(self.phase_in[i]) - self._halting(self.phase_out[i])

    def _argmax_pressure(self, exclude: int | None = None) -> int:
        """Index of the highest-pressure phase; ties resolved by lowest index."""
        best_i, best_p = self.idx, None
        for i in range(len(self.cycle)):
            if exclude is not None and i == exclude:
                continue
            p = self._pressure(i)
            if best_p is None or p > best_p:
                best_i, best_p = i, p
        return best_i

    def decide(self) -> list[int]:
        """Return the canonical phase array to apply at this step."""
        if not self._resolved:
            self._resolve_lanes()

        self.green_time += self.step_length

        if self.green_time >= self.min_green:
            if self.green_time >= self.max_green:
                # Forced switch to the best alternative phase (anti-starvation).
                new_idx = self._argmax_pressure(exclude=self.idx)
                if new_idx != self.idx:
                    self.idx = new_idx
                    self.green_time = 0.0
            else:
                new_idx = self._argmax_pressure()
                if new_idx != self.idx:
                    self.idx = new_idx
                    self.green_time = 0.0

        return self.cycle[self.idx]
