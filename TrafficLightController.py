"""Canonical 27-slot (13 vehicular + 14 pedestrian) traffic light controller for SUMO/TraCI."""

from __future__ import annotations

import numpy as np
import traci
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# Canonical model constants
VEHICLE_SLOT_NAMES = [
    "North/Right", "North/Middle", "North/Left",
    "East/Right", "East/Middle", "East/Left",
    "South/Right", "South/Middle", "South/Left",
    "West/Right", "West/Middle", "West/Left",
    "North-South/Track"
]

PEDESTRIAN_SLOT_NAMES = [
    "North/Right", "North/Middle+Left", "North/Track", "North/Departing",
    "East/Right", "East/Middle+Left", "East/Departing",
    "South/Right", "South/Middle+Left", "South/Track", "South/Departing",
    "West/Right", "West/Middle+Left", "West/Departing"
]

VEHICLE_SLOT_COUNT    = len(VEHICLE_SLOT_NAMES)      # 13
PEDESTRIAN_SLOT_COUNT = len(PEDESTRIAN_SLOT_NAMES)   # 14
CANONICAL_SLOT_COUNT  = VEHICLE_SLOT_COUNT + PEDESTRIAN_SLOT_COUNT  # 27
PEDESTRIAN_OFFSET = VEHICLE_SLOT_COUNT


class SlotType(IntEnum):
    VEHICLE    = 0
    PEDESTRIAN = 1

class SignalState(IntEnum):
    RED = 0
    GREEN = 1
    YELLOW = 2
    ALL_RED = 3
    RED_YELLOW = 4


def canonical_index(name: str, slot_type: SlotType) -> int:
    if slot_type == SlotType.VEHICLE:
        return VEHICLE_SLOT_NAMES.index(name)
    else:
        return PEDESTRIAN_OFFSET + PEDESTRIAN_SLOT_NAMES.index(name)


@dataclass
class LaneSignalMapping:
    """Maps one canonical slot to its physical SUMO signals and observable lanes."""
    name:          str
    slot_type:     SlotType
    sumo_tls_indices:  list[int] = field(default_factory=list)
    sumo_lane_ids: list[str] = field(default_factory=list)
    active:        bool      = True
    max_capacity:  int       = 0

    @property
    def canonical_index(self) -> int:
        return canonical_index(self.name, self.slot_type)


@dataclass
class PhaseDefinition:
    """One collision-free phase: a length-27 binary vector (1=green) plus a label."""
    signals: list[int]
    label:   str = ""

    def __post_init__(self):
        if len(self.signals) != CANONICAL_SLOT_COUNT:
            raise ValueError(
                f"PhaseDefinition.signals must have exactly {CANONICAL_SLOT_COUNT} "
                f"elements (got {len(self.signals)}): "
                f"{VEHICLE_SLOT_COUNT} vehicular + {PEDESTRIAN_SLOT_COUNT} pedestrian."
            )
        if any(s not in (0, 1) for s in self.signals):
            raise ValueError("PhaseDefinition.signals must contain only 0 or 1.")


# Transition timing configuration
@dataclass
class TransitionTiming:
    """Per-stage durations (steps) for the green -> yellow -> all-red -> red-yellow -> green sequence."""
    yellow_steps:              int = 3
    all_red_steps:             int = 1
    crosswalk_extra_red_steps: int = 3
    red_yellow_steps:          int = 2
    green_steps:               int = 0

class PhaseTracker:
    """Optional collector of phase selection counts and green durations for post-hoc analysis."""

    def __init__(self, phases: list[PhaseDefinition]) -> None:
        self._phases = phases
        self._labels: list[str] = [
            p.label if p.label else f"Phase_{i}" for i, p in enumerate(phases)
        ]
        self._counts: Counter = Counter()

        # Consecutive phase streaks
        self._last_phase: int | None = None
        self._current_streak: int = 0
        self._streaks: dict[str, list[int]] = {label: [] for label in self._labels}

        # Per-signal continuous green durations
        self._current_signal_streaks: list[int] = [0] * CANONICAL_SLOT_COUNT
        self._signal_streaks: list[list[int]] = [[] for _ in range(CANONICAL_SLOT_COUNT)]
        self._slot_names: list[str] = self._generate_slot_names()
        
        self._distance_sum: float = 0.0
        self._distance_count: int = 0

    def _generate_slot_names(self) -> list[str]:
        names = []
        for name in VEHICLE_SLOT_NAMES:
            names.append(f"Veh {name}")
        for name in PEDESTRIAN_SLOT_NAMES:
            names.append(f"Ped {name}")
        return names

    def record(self, phase_index: int, priority_scores: np.ndarray | None = None) -> None:
        """Record one phase selection by its index in valid_phases."""
        label = self._labels[phase_index]
        self._counts[label] += 1

        if priority_scores is not None:
            signals = self._phases[phase_index].signals
            dist = float(np.sum(np.abs(priority_scores - np.array(signals))))
            self._distance_sum += dist
            self._distance_count += 1

        # Full phase streaks
        if self._last_phase == phase_index:
            self._current_streak += 1
        else:
            if self._last_phase is not None:
                last_label = self._labels[self._last_phase]
                self._streaks[last_label].append(self._current_streak)
            self._current_streak = 1
            self._last_phase = phase_index

        # Per-signal streaks
        signals = self._phases[phase_index].signals
        for i, is_green in enumerate(signals):
            if is_green:
                self._current_signal_streaks[i] += 1
            else:
                if self._current_signal_streaks[i] > 0:
                    self._signal_streaks[i].append(self._current_signal_streaks[i])
                    self._current_signal_streaks[i] = 0

    def finalize_episode(self) -> None:
        """Save the final streak at episode end."""
        if self._last_phase is not None and self._current_streak > 0:
            last_label = self._labels[self._last_phase]
            self._streaks[last_label].append(self._current_streak)
            
        self._last_phase = None
        self._current_streak = 0
        
        for i in range(CANONICAL_SLOT_COUNT):
            if self._current_signal_streaks[i] > 0:
                self._signal_streaks[i].append(self._current_signal_streaks[i])
                self._current_signal_streaks[i] = 0

    def reset(self) -> None:
        """Clear all recorded counts and streak histories."""
        self._counts.clear()
        self._last_phase = None
        self._current_streak = 0
        for label in self._labels:
            self._streaks[label] = []
        
        self._current_signal_streaks = [0] * CANONICAL_SLOT_COUNT
        self._signal_streaks = [[] for _ in range(CANONICAL_SLOT_COUNT)]
        
        self._distance_sum = 0.0
        self._distance_count = 0

    @property
    def total_selections(self) -> int:
        return sum(self._counts.values())

    def get_distribution(self) -> dict[str, int]:
        """Return a dict mapping phase label -> selection count, sorted by count."""
        return dict(self._counts.most_common())

    def print_summary(self, title: str = "Phase Selection Distribution") -> None:
        """Print a frequency table with duration stats to stdout."""
        self.finalize_episode()
        total = self.total_selections
        if total == 0:
            print(f"[PhaseTracker] {title}: no selections recorded.")
            return

        col_w = max(max(len(label) for label in self._labels) + 2, 20)
        print(f"\n{'=' * (col_w + 35)}")
        print(f"  {title}")
        print(f"  Total steps evaluated: {total}")
        if self._distance_count > 0:
            avg_dist = self._distance_sum / self._distance_count
            print(f"  Model Phase Prediction Avg L1 Distance: {avg_dist:.4f} / {CANONICAL_SLOT_COUNT}")
        print(f"{'=' * (col_w + 35)}")
        print(f"  {'Phase':<{col_w}} {'Count':>8} {'Share':>7} | {'Avg Dur':>7} {'Max Dur':>7}")
        print(f"  {'-' * (col_w + 33)}")
        for label, count in self._counts.most_common():
            share = count / total * 100

            phase_streaks = self._streaks.get(label, [])
            avg_dur = sum(phase_streaks) / len(phase_streaks) if phase_streaks else 0.0
            max_dur = max(phase_streaks) if phase_streaks else 0
            
            print(f"  {label:<{col_w}} {count:>8} {share:>6.1f}% | {avg_dur:>7.1f} {max_dur:>7}")
        print(f"{'=' * (col_w + 35)}\n")
        
        print(f"\n{'=' * 45}")
        print(f"  Individual Signal Green Durations")
        print(f"{'=' * 45}")
        print(f"  {'Signal Slot':<20} | {'Avg Dur':>7} {'Max Dur':>7}")
        print(f"  {'-' * 43}")
        for i in range(CANONICAL_SLOT_COUNT):
            s_streaks = self._signal_streaks[i]
            if not s_streaks:
                continue
            
            avg_green = sum(s_streaks) / len(s_streaks)
            max_green = max(s_streaks)
            print(f"  {self._slot_names[i]:<20} | {avg_green:>7.1f} {max_green:>7}")
        print(f"{'=' * 45}\n")


class TrafficLightController:
    """Controls a SUMO intersection in 27-slot canonical space, mapping to physical signals via a lane map."""

    def __init__(
        self,
        sumo_id:                 str,
        lane_map:                list[LaneSignalMapping],
        valid_phases:            list[PhaseDefinition],
        sumo_signal_count:       int,
        timing:                  Optional[TransitionTiming] = None,
        standard_cycle:          Optional[list[PhaseDefinition]] = None,
        use_static_fallback:     bool = False,
        tracker:                 Optional[PhaseTracker]     = None,
        verbose:                 bool = False,
        verbose_phase_selection: bool = False,
    ):
        self.sumo_id                 = sumo_id
        self.timing                  = timing or TransitionTiming()
        self.tracker                 = tracker
        self.verbose                 = verbose
        self.verbose_phase_selection = verbose_phase_selection
        self.sumo_signal_count       = sumo_signal_count
        
        self.use_static_fallback     = use_static_fallback
        self._standard_cycle         = standard_cycle or []
        self._static_cycle_active    = False
        self._static_cycle_start_time = 0.0
        self._next_static_cycle_time = 300.0  # 5 minutes

        self._validate_and_store_lane_map(lane_map)
        self._validate_and_store_phases(valid_phases)

        # Phase matrix for vectorised scoring: shape (n_phases, 27)
        self._phase_matrix = np.array(
            [p.signals for p in self._valid_phases], dtype=np.float32
        )

        self._current_phase: list[int] = [0] * CANONICAL_SLOT_COUNT

        # Per-slot async state machines (tick_async)
        self._async_states: list[SignalState] = [SignalState.RED] * CANONICAL_SLOT_COUNT
        self._async_targets: list[int] = [0] * CANONICAL_SLOT_COUNT
        self._async_timers: list[int] = [0] * CANONICAL_SLOT_COUNT
        self._async_locked: list[bool] = [False] * CANONICAL_SLOT_COUNT

    def _validate_and_store_lane_map(self, lane_map: list[LaneSignalMapping]) -> None:
        """Validate the lane map (count, unique indices, in-range SUMO indices) and build lookups."""
        if len(lane_map) != CANONICAL_SLOT_COUNT:
            raise ValueError(
                f"lane_map must contain exactly {CANONICAL_SLOT_COUNT} entries "
                f"({VEHICLE_SLOT_COUNT} vehicular + {PEDESTRIAN_SLOT_COUNT} pedestrian), "
                f"got {len(lane_map)}."
            )

        canonical_seen: set[int] = set()

        for mapping in lane_map:
            cidx = mapping.canonical_index
            if cidx in canonical_seen:
                raise ValueError(
                    f"Duplicate canonical index {cidx} "
                    f"({mapping.slot_type.name} {mapping.name}) "
                    f"in lane_map."
                )
            canonical_seen.add(cidx)

            for sidx in mapping.sumo_tls_indices:
                if not (0 <= sidx < self.sumo_signal_count):
                    raise ValueError(
                        f"SUMO index {sidx} for "
                        f"{mapping.slot_type.name} {mapping.name} "
                        f"is out of range [0, {self.sumo_signal_count})."
                    )

        self._lane_map: list[LaneSignalMapping] = sorted(
            lane_map, key=lambda m: m.canonical_index
        )
        self._canonical_to_mapping: dict[int, LaneSignalMapping] = {
            m.canonical_index: m for m in self._lane_map
        }
        self._active_mask: list[bool] = [
            self._canonical_to_mapping[i].active for i in range(CANONICAL_SLOT_COUNT)
        ]

    def _validate_and_store_phases(self, valid_phases: list[PhaseDefinition]) -> None:
        if not valid_phases:
            raise ValueError("valid_phases must contain at least one PhaseDefinition.")
        self._valid_phases = valid_phases

    def get_vehicle_slots(self) -> list[LaneSignalMapping]:
        """Return the 13 vehicular slot mappings in canonical order."""
        return [
            self._canonical_to_mapping[i]
            for i in range(VEHICLE_SLOT_COUNT)
        ]

    def get_pedestrian_slots(self) -> list[LaneSignalMapping]:
        """Return the 14 pedestrian slot mappings in canonical order."""
        return [
            self._canonical_to_mapping[i]
            for i in range(PEDESTRIAN_OFFSET, CANONICAL_SLOT_COUNT)
        ]

    def get_best_phase(
        self,
        priority_scores: np.ndarray,
        scoring: str = "linear",
        constrained_targets: Optional[list[int]] = None,
        constrained_mask: Optional[list[bool]] = None,
    ) -> list[int]:
        """Pick the valid phase best matching the priority scores, honouring any locked-signal constraints."""
        scores        = np.asarray(priority_scores, dtype=np.float32)
        active        = np.array(self._active_mask,  dtype=np.float32)
        masked_scores = scores * active

        # Filter phases by constraints if provided
        valid_indices = []
        for i, p in enumerate(self._valid_phases):
            valid = True
            if constrained_mask is not None and constrained_targets is not None:
                for j, (mask, target) in enumerate(zip(constrained_mask, constrained_targets)):
                    if mask and p.signals[j] != target:
                        valid = False
                        break
            if valid:
                valid_indices.append(i)

        if not valid_indices:
            # Fall back to all phases if constraints leave none
            valid_indices = list(range(len(self._valid_phases)))

        valid_phase_matrix = np.array([self._valid_phases[i].signals for i in valid_indices], dtype=np.float32)

        if scoring == "linear":
            phase_scores = valid_phase_matrix @ masked_scores
        elif scoring == "sigmoid":
            phase_scores = np.sum(
                1.0 / (1.0 + np.exp(-10.0 * valid_phase_matrix * masked_scores + 5.0)),
                axis=1,
            )
        elif scoring == "symmetric":
             green_match = valid_phase_matrix * scores
             red_match = (1.0 - valid_phase_matrix) * (1.0 - scores)
             phase_scores = np.sum((green_match + red_match) * active, axis=1)
        else:
            raise ValueError(f"Unknown scoring method: '{scoring}'.")

        best_local_index = int(np.argmax(phase_scores))
        best_global_index = valid_indices[best_local_index]

        if self.verbose_phase_selection:
            print(f"Model wants:\n{priority_scores}")
            pass

        if self.tracker is not None:
            self.tracker.record(best_global_index, priority_scores=scores)

        return list(self._valid_phases[best_global_index].signals)

    def _print_phase_ranking(self, phase_scores: np.ndarray, best_index: int) -> None:
        """Print a ranked table of phases and scores to stdout."""
        ranked = sorted(enumerate(phase_scores), key=lambda x: x[1], reverse=True)
        col_w  = max(
            len(p.label if p.label else f"Phase_{i}")
            for i, p in enumerate(self._valid_phases)
        ) + 2
        print(f"\n  [{self.sumo_id}] Phase selection ranking:")
        print(f"  {'Phase':<{col_w}} {'Score':>8}")
        print(f"  {'-' * (col_w + 12)}")
        for idx, score in ranked:
            label    = self._valid_phases[idx].label or f"Phase_{idx}"
            marker   = " <-- selected" if idx == best_index else ""
            print(f"  {label:<{col_w}} {score:>8.4f}{marker}")
        print()

    def _canonical_to_sumo_string(self, signal_map: dict[int, str]) -> str:
        """Build the SUMO signal string from a canonical-index -> character map (uncovered indices default to 'r')."""
        chars = ['r'] * self.sumo_signal_count
        for cidx, mapping in self._canonical_to_mapping.items():
            char = signal_map.get(cidx, 'r')
            for sidx in mapping.sumo_tls_indices:
                chars[sidx] = char
        return ''.join(chars)

    def _build_signal_map(self, new_phase: list[int], stage: str) -> dict[int, str]:
        """Map each canonical slot to its SUMO character for a transition stage; pedestrian slots skip y/u."""
        signal_map: dict[int, str] = {}

        for cidx in range(CANONICAL_SLOT_COUNT):
            was_green  = self._current_phase[cidx] == 1
            will_green = new_phase[cidx] == 1
            is_pedestrian = cidx >= PEDESTRIAN_OFFSET

            if stage == "yellow":
                if   was_green and     will_green:  signal_map[cidx] = 'G'
                elif was_green and not will_green:  signal_map[cidx] = 'r' if is_pedestrian else 'y'
                else:                               signal_map[cidx] = 'r'

            elif stage == "all_red":
                signal_map[cidx] = 'G' if (was_green and will_green) else 'r'

            elif stage == "red_yellow":
                if       was_green and     will_green:  signal_map[cidx] = 'G'
                elif not was_green and     will_green:  signal_map[cidx] = 'r' if is_pedestrian else 'u'
                else:                                   signal_map[cidx] = 'r'

            elif stage == "green":
                signal_map[cidx] = 'G' if will_green else 'r'

            else:
                raise ValueError(f"Unknown transition stage: '{stage}'.")

        return signal_map

    def _any_pedestrian_slot_terminates(self, new_phase: list[int]) -> bool:
        """True if any active pedestrian slot goes green -> red (triggers extended all-red)."""
        for cidx in range(PEDESTRIAN_OFFSET, CANONICAL_SLOT_COUNT):
            if self._active_mask[cidx] and self._current_phase[cidx] == 1 and new_phase[cidx] == 0:
                return True
        return False

    @staticmethod
    def _count_waiting_vehicles() -> int:
        """Count simulation-wide vehicles with speed < 0.1 m/s."""
        return sum(
            1 for v in traci.vehicle.getIDList()
            if traci.vehicle.getSpeed(v) < 0.1
        )

    def _step(self, label: str, transition_callback=None) -> int:
        """Advance the simulation one step and return the waiting vehicle count."""
        if self.verbose:
            print(f"  [{self.sumo_id}] {label}")
        traci.simulationStep()
        if transition_callback:
            return transition_callback()
        return self._count_waiting_vehicles()

    def _run_stage(self, stage: str, new_phase: list[int], steps: int, transition_callback=None) -> int:
        """Apply the stage signal string, advance `steps` steps, return cumulative waiting count."""
        traci.trafficlight.setRedYellowGreenState(
            self.sumo_id,
            self._canonical_to_sumo_string(self._build_signal_map(new_phase, stage))
        )

        if steps <= 0:
            return 0

        return sum(self._step(stage, transition_callback) for _ in range(steps))

    def tick_async(
        self,
        priority_scores: np.ndarray,
        scoring: str = "linear"
    ) -> tuple[int, list[int]]:
        """Step-by-step transition with an independent state machine per canonical slot."""
        new_phase = self.get_best_phase(
            priority_scores, 
            scoring=scoring, 
            constrained_targets=self._async_targets, 
            constrained_mask=self._async_locked
        )

        light_changes = [0] * CANONICAL_SLOT_COUNT

        # Check if any pedestrian slot is terminating in this tick
        any_ped_terminates = False
        for cidx in range(PEDESTRIAN_OFFSET, CANONICAL_SLOT_COUNT):
            if self._active_mask[cidx] and self._async_states[cidx] == SignalState.GREEN and new_phase[cidx] == 0:
                any_ped_terminates = True

        ped_extra_red = self.timing.crosswalk_extra_red_steps if any_ped_terminates else 0

        # Update targets for unlocked signals
        for i in range(CANONICAL_SLOT_COUNT):
            if not self._async_locked[i]:
                if self._async_states[i] == SignalState.GREEN and new_phase[i] == 0:
                    self._async_targets[i] = 0
                    self._async_locked[i] = True
                    self._async_states[i] = SignalState.YELLOW
                    self._async_timers[i] = self.timing.yellow_steps
                    light_changes[i] = 1
                elif self._async_states[i] == SignalState.RED and new_phase[i] == 1:
                    self._async_targets[i] = 1
                    self._async_locked[i] = True
                    self._async_states[i] = SignalState.RED
                    # Wait for conflicting signals to clear
                    self._async_timers[i] = self.timing.yellow_steps + self.timing.all_red_steps + ped_extra_red
                    light_changes[i] = 1

        # Process active transitions (countdown timers)
        for i in range(CANONICAL_SLOT_COUNT):
            if self._async_locked[i]:
                if self._async_timers[i] <= 0:
                    if self._async_states[i] == SignalState.YELLOW:
                        self._async_states[i] = SignalState.RED
                        self._async_timers[i] = self.timing.all_red_steps + ped_extra_red + self.timing.red_yellow_steps + self.timing.green_steps
                    elif self._async_states[i] == SignalState.RED:
                        if self._async_targets[i] == 0:
                            self._async_locked[i] = False
                        elif self._async_targets[i] == 1:
                            self._async_states[i] = SignalState.RED_YELLOW
                            self._async_timers[i] = self.timing.red_yellow_steps
                    elif self._async_states[i] == SignalState.RED_YELLOW:
                        self._async_states[i] = SignalState.GREEN
                        self._async_timers[i] = self.timing.green_steps
                    elif self._async_states[i] == SignalState.GREEN:
                        self._async_locked[i] = False

                # Decrement timer for this tick
                if self._async_timers[i] > 0:
                    self._async_timers[i] -= 1

        # Build SUMO string from async state
        signal_map = {}
        for cidx in range(CANONICAL_SLOT_COUNT):
            state = self._async_states[cidx]
            is_ped = cidx >= PEDESTRIAN_OFFSET
            if state == SignalState.GREEN: signal_map[cidx] = 'G'
            elif state == SignalState.YELLOW: signal_map[cidx] = 'r' if is_ped else 'y'
            elif state == SignalState.RED_YELLOW: signal_map[cidx] = 'r' if is_ped else 'u'
            elif state == SignalState.RED: signal_map[cidx] = 'r'

        sumo_string = self._canonical_to_sumo_string(signal_map)
        traci.trafficlight.setRedYellowGreenState(self.sumo_id, sumo_string)

        wait_time = self._step("async_tick")

        return wait_time, light_changes

    def transition_to(
        self,
        priority_scores: np.ndarray,
        transition_callback = None,
        scoring: str = "linear",
    ) -> tuple[int, list[int]]:
        """Pick the best phase and run the full yellow/all-red/red-yellow/green transition; return (wait_time, light_changes)."""

        # Determine new phase
        current_time = traci.simulation.getTime()
        
        if self.use_static_fallback and self._standard_cycle:
            if not self._static_cycle_active and current_time >= self._next_static_cycle_time:
                self._static_cycle_active = True
                self._static_cycle_start_time = current_time
                if self.verbose:
                    print(f"[{self.sumo_id}] Starting static fallback cycle at {current_time}s")

            if self._static_cycle_active:
                time_in_cycle = current_time - self._static_cycle_start_time
                phase_idx = int(time_in_cycle // 15)
                
                if phase_idx >= len(self._standard_cycle):
                    self._static_cycle_active = False
                    self._next_static_cycle_time = current_time + 300.0  # 5 minutes from end
                    if self.verbose:
                        print(f"[{self.sumo_id}] Ended static fallback cycle")
                    new_phase = self.get_best_phase(priority_scores, scoring=scoring)
                else:
                    new_phase = list(self._standard_cycle[phase_idx].signals)
            else:
                new_phase = self.get_best_phase(priority_scores, scoring=scoring)
        else:
            new_phase = self.get_best_phase(priority_scores, scoring=scoring)
            
        phase_changed = new_phase != self._current_phase

        if not phase_changed:
            if self.verbose:
                print(f"[{self.sumo_id}] Transition: NONE")
            return 0, [0] * CANONICAL_SLOT_COUNT

        if self.verbose:
            print(f"[{self.sumo_id}] Transition: {self._current_phase} -> {new_phase}")
            print(f"\t\t\t\tPedestrian slot ends: {self._any_pedestrian_slot_terminates(new_phase)}")
        
        # Yellow
        wait_time  = self._run_stage("yellow", new_phase, self.timing.yellow_steps, transition_callback)
        # All-red
        wait_time += self._run_stage("all_red", new_phase, self.timing.all_red_steps, transition_callback)
        # Pedestrian all-red extension
        wait_time += self._run_stage("all_red", new_phase, self.timing.crosswalk_extra_red_steps if self._any_pedestrian_slot_terminates(new_phase) else 0, transition_callback)
        # Red-yellow
        wait_time += self._run_stage("red_yellow", new_phase, self.timing.red_yellow_steps, transition_callback)
        # Green
        wait_time += self._run_stage("green", new_phase, self.timing.green_steps, transition_callback)

        # Update light_changes map and current phase
        light_changes = [
            1 if self._current_phase[i] != new_phase[i] else 0
            for i in range(CANONICAL_SLOT_COUNT)
        ]
        self._current_phase = new_phase

        return wait_time, light_changes

    @property
    def current_phase(self) -> list[int]:
        """Current canonical phase as a 27-element binary list (copy)."""
        return list(self._current_phase)

    @property
    def active_mask(self) -> list[bool]:
        """Boolean mask of active canonical slots (read-only copy)."""
        return list(self._active_mask)

    def get_sumo_signal_string(self) -> str:
        """Return the live SUMO signal string for this junction."""
        return traci.trafficlight.getRedYellowGreenState(self.sumo_id)

    def __repr__(self) -> str:
        active_vehicle    = sum(self._active_mask[:PEDESTRIAN_OFFSET])
        active_pedestrian = sum(self._active_mask[PEDESTRIAN_OFFSET:])
        return (
            f"TrafficLightController(sumo_id={self.sumo_id!r}, "
            f"active_vehicle={active_vehicle}/{VEHICLE_SLOT_COUNT}, "
            f"active_pedestrian={active_pedestrian}/{PEDESTRIAN_SLOT_COUNT}, "
            f"valid_phases={len(self._valid_phases)}, "
            f"sumo_signals={self.sumo_signal_count})"
        )
