"""Training env + curriculum for the SUMO intersection RL agent (147-dim obs, 27-dim action)."""
# GPU env setup
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTHONUNBUFFERED"] = "1"

# Ablation switch read from environment (default keeps original behaviour).
_REWARD_MODE = os.environ.get("REWARD_MODE", "full").lower()

import time
from dataclasses import dataclass, field

import numpy as np
import gymnasium
from gymnasium import spaces
# Import before libsumo so pyexpat's DLL loads first (Windows)
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

import sys as _sys
# Optional in-process backend: USE_LIBSUMO=1 swaps traci for libsumo (faster, identical results)
if os.environ.get("USE_LIBSUMO", "0") == "1":
    try:
        import libsumo as _libsumo
        if not hasattr(_libsumo, "exceptions"):
            import types as _types
            _ex = _types.ModuleType("traci.exceptions")
            _ex.TraCIException = getattr(_libsumo, "TraCIException", Exception)
            _ex.FatalTraCIError = getattr(_libsumo, "FatalTraCIError", Exception)
            _libsumo.exceptions = _ex
            _sys.modules["traci.exceptions"] = _ex
        _sys.modules["traci"] = _libsumo
        print("[backend] Using libsumo (in-process)")
    except Exception as _e:
        print(f"[backend] libsumo unavailable ({_e}); falling back to traci")
import traci

from TrafficLightController import (
    TrafficLightController,
    LaneSignalMapping,
    PhaseTracker,
    CANONICAL_SLOT_COUNT,
    VEHICLE_SLOT_COUNT,
    PEDESTRIAN_SLOT_COUNT,
    PEDESTRIAN_OFFSET,
)
from IntersectionRoutes import RouteSlot
from intersection_config import load_intersection_config
import re

# Environment constants
EPISODE_DURATION = 2000
BUS_WEIGHT = 2.0
TRAM_WEIGHT = 3.0

# Observation layout
VEHICLE_FEATURES_PER_SLOT = 7
PEDESTRIAN_FEATURES_PER_SLOT = 4
OBS_SIZE = (
    VEHICLE_SLOT_COUNT * VEHICLE_FEATURES_PER_SLOT + PEDESTRIAN_SLOT_COUNT * PEDESTRIAN_FEATURES_PER_SLOT
)                                              # 91 + 56 = 147


@dataclass
class CurriculumLevel:
    """One difficulty tier: a name, a list of {yaml, sumocfg} configs, and an upgrade step threshold."""
    name:           str
    configs:        list[dict]
    step_threshold: int = 0


@dataclass
class CurriculumSchedule:
    """Ordered CurriculumLevels: configs rotate round-robin per reset, levels upgrade on step thresholds."""
    levels: list[CurriculumLevel] = field(default_factory=list)

    def __post_init__(self):
        if not self.levels:
            raise ValueError("CurriculumSchedule must contain at least one CurriculumLevel.")

    def current_config(self, level_index: int, config_index: int) -> dict:
        level_configs = self.levels[level_index].configs
        return level_configs[config_index % len(level_configs)]

    def should_upgrade(self, level_index: int, global_step: int) -> bool:
        level = self.levels[level_index]
        return (
            level.step_threshold > 0
            and global_step >= level.step_threshold
            and level_index < len(self.levels) - 1
        )

    def next_level_index(self, level_index: int) -> int:
        return min(level_index + 1, len(self.levels) - 1)


class CurriculumCallback(BaseCallback):
    """SB3 callback that upgrades the curriculum level based on the global step count."""

    def __init__(self, curriculum_schedule: CurriculumSchedule, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum = curriculum_schedule
        self.level_index = 0

    def _on_step(self) -> bool:
        if self.curriculum is None:
            return True

        global_step = self.num_timesteps

        if self.curriculum.should_upgrade(self.level_index, global_step):
            new_index = self.curriculum.next_level_index(self.level_index)
            old_name = self.curriculum.levels[self.level_index].name
            self.level_index = new_index

            # Broadcast to all parallel envs
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method("set_curriculum_level", new_index)
                
            if self.verbose:
                print(
                    f"\n[Curriculum] Step {global_step}: "
                    f"'{old_name}' -> "
                    f"'{self.curriculum.levels[new_index].name}'\n"
                )
        return True


class SumoIntersectionEnv(gymnasium.Env):
    """Gymnasium env for one SUMO intersection; spaces are derived from the controller's lane map."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        controller:   TrafficLightController,
        bounds:       dict,
        curriculum:   CurriculumSchedule | None   = None,
        sumo_cmd:     list[str] | None            = None,
        gui:          bool                        = False,
        route_slots:  list[RouteSlot] | None      = None,
        use_static_fallback: bool                 = False,
    ):
        super().__init__()

        if curriculum is None and sumo_cmd is None:
            raise ValueError("Provide either a CurriculumSchedule or a fixed sumo_cmd.")

        self.controller      = controller
        self.curriculum      = curriculum
        self.gui             = gui
        self._fixed_sumo_cmd = sumo_cmd
        self.use_static_fallback = use_static_fallback
        self.controller.use_static_fallback = self.use_static_fallback

        # Normalisation bounds strictly for this intersection configuration
        self.max_vehicles_in_lane = bounds.get("max_vehicles_in_lane", 20)
        self.max_wait_per_vehicle = bounds.get("max_wait_per_vehicle", 90.0)
        self.max_light_age        = bounds.get("max_light_age", 60.0)
        self.max_wait_in_lane     = self.max_vehicles_in_lane * self.max_wait_per_vehicle
        
        # Track the active yaml so we don't unnecessarily reload if it hasn't changed.
        self._current_yaml_config = None

        # One route slot per vehicular canonical slot; None falls back to single-lane distance
        self._route_slots: list[RouteSlot] | None = route_slots

        # Filled in reset() once TraCI is live (edge lengths need a connection)
        self._slot_total_lengths: list[float] | None = None

        # Cache slot lists from controller (constant for the lifetime of the env)
        self._vehicle_slots:    list[LaneSignalMapping] = controller.get_vehicle_slots()
        self._pedestrian_slots: list[LaneSignalMapping] = controller.get_pedestrian_slots()

        # Dynamic max throughput per step for Trip Quality reward (roughly 1/3 of active vehicular slots)
        active_veh_slots = sum(1 for slot in self._vehicle_slots if slot.active)
        self.max_throughput_per_step = max(1.0, float(active_veh_slots) / 3.0)

        # Prepare list of incoming edges (to differentiate from outgoing for reward)
        self.incoming_edges = set()
        if self._route_slots:
            for slot in self._route_slots:
                for fe in slot.feed_edges:
                    self.incoming_edges.add(fe.edge_id)
                if hasattr(slot, "approaches") and slot.approaches:
                    for approach_chain in slot.approaches:
                        for fe in approach_chain:
                            self.incoming_edges.add(fe.edge_id)

        # Spaces
        self.action_space      = spaces.Box(low=0.0, high=1.0, shape=(CANONICAL_SLOT_COUNT,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,),             dtype=np.float32)

        # Curriculum state
        self.curriculum_level_index:  int = 0
        self.curriculum_config_index: int = 0
        self._ablate_no_lead = os.environ.get("ABLATE_NO_LEAD") == "1"

    def set_curriculum_level(self, level_index: int):
        """Called by CurriculumCallback to update the difficulty tier across vectorized envs."""
        self.curriculum_level_index = level_index
        self.curriculum_config_index = 0

        # Episode statistics (reset each episode)
        self.total_cumulative_wait_time: float = 0.0
        self.max_cumulative_wait_time:   float = 0.0
        self.current_max_wait_time:      float = 0.0
        self.max_wait_time_sec:          float = 0.0
        self.vehicles_seen:              set   = set()
        self.priority_vehicles_seen:     set   = set()
        self.n_moving:                   int   = 0
        self.n_waiting:                  int   = 0
        self.prev_total_accumulated_wait:float = 0.0

        # Per-slot light age counters
        self._time_since_light_change = np.zeros(CANONICAL_SLOT_COUNT,  dtype=np.float32)
        self._light_changes           = np.zeros(CANONICAL_SLOT_COUNT,  dtype=np.int32)

    def _update_intersection_config(self, yaml_config: str):
        """Update intersection topology only if the curriculum provided a new yaml."""
        if self._current_yaml_config == yaml_config:
            return

        print(f"[{self.__class__.__name__}] Loading new intersection config: {yaml_config}")

        self.controller, self._route_slots, bounds = load_intersection_config(yaml_config)
        self.controller.use_static_fallback = self.use_static_fallback
        self._current_yaml_config = yaml_config

        # Update dependent bounds
        self.max_vehicles_in_lane = bounds.get("max_vehicles_in_lane", 20)
        self.max_wait_per_vehicle = bounds.get("max_wait_per_vehicle", 90.0)
        self.max_light_age        = bounds.get("max_light_age", 60.0)
        self.max_wait_in_lane     = self.max_vehicles_in_lane * self.max_wait_per_vehicle

        # Reset lengths so they recalculate after TraCI loads
        self._slot_total_lengths = None

        # Re-cache slot dependencies
        self._vehicle_slots      = self.controller.get_vehicle_slots()
        self._pedestrian_slots   = self.controller.get_pedestrian_slots()

        # Update dynamic throughput max
        active_veh_slots = sum(1 for slot in self._vehicle_slots if slot.active)
        self.max_throughput_per_step = max(1.0, float(active_veh_slots) / 3.0)

        # Precompute incoming edges for reward logic
        self.incoming_edges = set()
        if self._route_slots:
            for slot in self._route_slots:
                for fe in slot.feed_edges:
                    self.incoming_edges.add(fe.edge_id)
                if hasattr(slot, "approaches") and slot.approaches:
                    for approach_chain in slot.approaches:
                        for fe in approach_chain:
                            self.incoming_edges.add(fe.edge_id)

    def _build_sumo_cmd(self) -> list[str]:
        if self.curriculum is not None:
            config_dict = self.curriculum.current_config(
                self.curriculum_level_index,
                self.curriculum_config_index,
            )
            
            yaml_config = config_dict.get("yaml", None)
            if yaml_config:
                self._update_intersection_config(yaml_config)
            
            # Default to the yaml's configured sumocfg_path if not explicitly provided
            _, _, default_bounds = load_intersection_config(yaml_config) if yaml_config else (None, None, {})
            sumocfg = config_dict.get("sumocfg", default_bounds.get("sumocfg_path", ""))

            self.curriculum_config_index += 1   # advance round-robin for next reset
            binary = "sumo-gui" if self.gui else "sumo"
            cmd = [
                binary, "-c", sumocfg,
                "--start",
                "--quit-on-end",
                "--step-length", "1.0",
                "--time-to-teleport", "-1",
                "--waiting-time-memory", f"{EPISODE_DURATION}",
                "--random",
            ]
        else:
            cmd = self._fixed_sumo_cmd[:]

        if self.gui and "--delay" not in cmd:
            cmd.extend(["--delay", "400"])

        return cmd

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        traci.start(self._build_sumo_cmd())

        # Compute slot lengths once TraCI is live (idempotent across episodes)
        if self._route_slots is not None and self._slot_total_lengths is None:
            self._slot_total_lengths = [s.total_length() for s in self._route_slots]

        self.total_cumulative_wait_time = 0.0
        self.max_cumulative_wait_time   = 0.0
        self.current_max_wait_time      = 0.0
        self.max_wait_time_sec          = 0.0
        self.vehicles_seen              = set()
        self.priority_vehicles_seen     = {}
        self.passed_vehicles            = set()
        self.n_moving                   = 0
        self.n_waiting                  = 0
        self.prev_total_accumulated_wait = 0.0
        self.prev_arrived               = 0
        self.current_arrived            = 0
        self._time_since_light_change   = np.zeros(CANONICAL_SLOT_COUNT, dtype=np.float32)
        self._light_changes             = np.zeros(CANONICAL_SLOT_COUNT, dtype=np.int32)
        self.arrived_during_transition  = set()

        return self._build_observation(), {}

    def step(self, action: np.ndarray):
        start_time = traci.simulation.getTime()
        
        def transition_callback():
            self.arrived_during_transition.update(traci.simulation.getArrivedIDList())
            waiting = 0
            for vid in traci.vehicle.getIDList():
                edge_id = traci.vehicle.getRoadID(vid)
                if (edge_id in self.incoming_edges):
                    if traci.vehicle.getSpeed(vid) < 0.1:
                        waiting += 1
            self.vehicles_seen.update(traci.vehicle.getIDList())
            return waiting

        self.arrived_during_transition.clear()
        transition_wait, light_changes_list = self.controller.transition_to(action, transition_callback)
        self._light_changes = np.array(light_changes_list, dtype=np.int32)

        traci.simulationStep()
        self.arrived_during_transition.update(traci.simulation.getArrivedIDList())
        
        sim_steps_passed = max(1.0, traci.simulation.getTime() - start_time)

        waiting = 0
        for vid in traci.vehicle.getIDList():
            edge_id = traci.vehicle.getRoadID(vid)
            if (edge_id in self.incoming_edges):
                if traci.vehicle.getSpeed(vid) < 0.1:
                    waiting += 1
                    
        self.n_waiting = waiting
        self.n_moving  = len(traci.vehicle.getIDList()) - self.n_waiting

        self.total_cumulative_wait_time += self.n_waiting + transition_wait
        
        prev_seen = len(self.vehicles_seen)
        self.vehicles_seen.update(traci.vehicle.getIDList())
        self.current_arrived = len(self.vehicles_seen) - prev_seen

        reward = self._compute_reward(sim_steps_passed)

        # Done at time limit or when the network is empty
        done = (
            traci.simulation.getTime() >= EPISODE_DURATION 
            or traci.simulation.getMinExpectedNumber() <= 0
        )

        if done:
            reward = self._compute_terminal_reward()
            self._log_episode_summary(reward)

        return self._build_observation(), reward, done, False, {}

    def _build_observation(self) -> np.ndarray:
        """Build the 147-element observation (91 vehicular + 56 pedestrian features)."""
        self.current_max_wait_time = 0.0
        signal_state = traci.trafficlight.getRedYellowGreenState(self.controller.sumo_id)
        obs_blocks: list[list[float]] = []
        present_ids = set(traci.vehicle.getIDList())  # vehicles currently in the network

        # ---- Vehicular slots ----------------------------------------
        for slot_idx, slot in enumerate(self._vehicle_slots):

            if not slot.active:
                obs_blocks.append([0.0] * VEHICLE_FEATURES_PER_SLOT)
                continue

            route    = self._route_slots[slot_idx]  if self._route_slots  else None
            norm_len = self._slot_total_lengths[slot_idx] if self._slot_total_lengths else None
            lane_ids = route.all_lane_ids() if route else slot.sumo_lane_ids

            total_vehicles    = 0
            total_priority    = 0
            min_dist_to_stop  = float("inf")   # raw metres; normalised at the end
            total_wait_sum    = 0.0
            max_wait_max      = 0.0

            for lane_id in lane_ids:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                total_vehicles += len(vehicle_ids)

                for veh_id in vehicle_ids:
                    if veh_id not in present_ids:
                        continue  # vehicle left the network between steps
                    wait    = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    v_class = traci.vehicle.getVehicleClass(veh_id)
                    pos     = traci.vehicle.getLanePosition(veh_id)
                    max_wait_max = max(max_wait_max, float(wait))
                    if v_class in ("bus", "tram"):
                        total_priority += 1
                    if route is not None:
                        dist = route.distance_to_stop_line(lane_id, pos)
                    else:
                        # Fallback: distance within single lane only
                        dist = traci.lane.getLength(lane_id) - pos
                    min_dist_to_stop = min(min_dist_to_stop, dist)

                total_wait_sum += traci.lane.getWaitingTime(lane_id)

            # Normalise lead distance by total slot length (or single lane length)
            if min_dist_to_stop == float("inf"):
                lead_distance_norm = 1.0   # no vehicle present
            elif norm_len and norm_len > 0:
                lead_distance_norm = min(min_dist_to_stop / norm_len, 1.0)
            else:
                lead_distance_norm = 1.0

            if self._ablate_no_lead:
                lead_distance_norm = 1.0  # ablation: mask the lead-distance state feature

            # Dynamic capacity based on the custom YAML capacity if present, otherwise fallback
            num_lanes = len(lane_ids) if lane_ids else 1
            if slot.max_capacity > 0:
                slot_max_vehicles = float(slot.max_capacity)
            else:
                slot_max_vehicles = self.max_vehicles_in_lane * num_lanes
                
            slot_max_wait = self.max_wait_in_lane * num_lanes

            vehicle_count_norm = min(total_vehicles / slot_max_vehicles, 1.5) if slot_max_vehicles > 0 else 0.0
            wait_sum_norm      = min(total_wait_sum / slot_max_wait, 1.5) if slot_max_wait > 0 else 0.0
            wait_max_norm      = min(max_wait_max   / self.max_wait_per_vehicle, 1.5)
            priority_coeff     = float(total_priority) / total_vehicles if total_vehicles > 0 else 0.0

            # Current light state for this canonical vehicular slot
            current_light = float(self.controller.current_phase[slot_idx])

            # Light age: reset on change, increment on hold
            if self._light_changes[slot_idx]:
                self._time_since_light_change[slot_idx] = 0.0
            else:
                self._time_since_light_change[slot_idx] += 1.0
            light_age_norm = min(self._time_since_light_change[slot_idx] / self.max_light_age, 1.5)

            self.current_max_wait_time    = max(self.current_max_wait_time,    wait_max_norm)
            self.max_cumulative_wait_time = max(self.max_cumulative_wait_time, wait_max_norm)

            obs_blocks.append([
                current_light,
                light_age_norm,
                vehicle_count_norm,
                lead_distance_norm,
                wait_sum_norm,
                wait_max_norm,
                priority_coeff,
            ])

        # ---- Pedestrian slots ---------------------------------------
        for i, slot in enumerate(self._pedestrian_slots):
            slot_idx = i + PEDESTRIAN_OFFSET

            if not slot.active:
                obs_blocks.append([0.0] * PEDESTRIAN_FEATURES_PER_SLOT)
                continue

            # Route for pedestrians (index starts after vehicular slots)
            route = self._route_slots[slot_idx] if self._route_slots else None
            lane_ids = route.all_lane_ids() if route else slot.sumo_lane_ids

            # Binary presence: 1 if any person detected on any crossing lane
            person_present = 0.0
            max_wait_max = 0.0
            for lane_id in lane_ids:
                # Persons are queried per edge (traci.lane has no person accessor)
                edge_id = traci.lane.getEdgeID(lane_id)
                person_ids = traci.edge.getLastStepPersonIDs(edge_id)
                
                if person_ids:
                    person_present = 1.0
                    wait_times = np.array(
                        [traci.person.getWaitingTime(p) for p in person_ids],
                        dtype=np.float32,
                    )
                    if wait_times.size > 0:
                        max_wait_max = max(max_wait_max, float(np.max(wait_times)))

            wait_max_norm = min(max_wait_max / self.max_wait_per_vehicle, 1.5)

            # Current light state for this canonical pedestrian slot
            current_light = float(self.controller.current_phase[slot_idx])

            # Light age for pedestrian slot
            if self._light_changes[slot_idx]:
                self._time_since_light_change[slot_idx] = 0.0
            else:
                self._time_since_light_change[slot_idx] += 1.0
            light_age_norm = min(self._time_since_light_change[slot_idx] / self.max_light_age, 1.5)

            self.current_max_wait_time    = max(self.current_max_wait_time,    wait_max_norm)
            self.max_cumulative_wait_time = max(self.max_cumulative_wait_time, wait_max_norm)

            obs_blocks.append([
                float(current_light),
                float(light_age_norm),
                float(person_present),
                float(wait_max_norm),
            ])

        # Flatten manually (blocks are inhomogeneous)
        flat_obs = []
        for block in obs_blocks:
            flat_obs.extend(block)

        return np.array(flat_obs, dtype=np.float32)

    def _compute_reward(self, sim_steps_passed: float = 1.0) -> float:
        """Per-step reward: penalise approaching/standing vehicles, reward newly passed ones, plus a pedestrian term."""
        current_passed = set()
        standing_penalty = 0.0
        
        # New counters for the moving vs non-moving ratio
        approaching_moving = 0.0
        weighted_approaching_moving = 0.0
        total_standing = 0.0

        active_vehicles = traci.vehicle.getIDList()
        present_ids = set(active_vehicles)
        current_max_wait = 0.0
        total_accumulated_wait = 0.0
        num_waiting_vehicles = 0
        edge_max_waits = {}

        for vid in active_vehicles:
            if vid not in self.priority_vehicles_seen:
                v_class = traci.vehicle.getVehicleClass(vid)
                if v_class == "bus":
                    self.priority_vehicles_seen[vid] = BUS_WEIGHT
                elif v_class == "tram":
                    self.priority_vehicles_seen[vid] = TRAM_WEIGHT
                else:
                    self.priority_vehicles_seen[vid] = 1.0  # Cache normal cars too to avoid repeated traci calls

            edge_id = traci.vehicle.getRoadID(vid)
            # An edge is 'approaching' if it's in our configured incoming slots
            is_approaching = (edge_id in self.incoming_edges)
            priority_mult = self.priority_vehicles_seen.get(vid, 1.0)
            
            try:
                speed = traci.vehicle.getSpeed(vid)
                if speed < 0.1:
                    total_standing += priority_mult
                else:
                    if is_approaching:
                        approaching_moving += priority_mult
                        try:
                            wait_time_moving = float(traci.vehicle.getAccumulatedWaitingTime(vid))
                        except traci.exceptions.TraCIException:
                            wait_time_moving = 0.0
                        wait_time_norm = wait_time_moving / self.max_wait_per_vehicle
                        weighted_approaching_moving += priority_mult * (1.0 + wait_time_norm)
            except traci.exceptions.TraCIException:
                pass
            
            if is_approaching and (vid not in current_passed):
                
                # ONLY count wait times for vehicles that haven't cleared the stop line
                try:
                    wait_time = float(traci.vehicle.getAccumulatedWaitingTime(vid))
                except traci.exceptions.TraCIException:
                    wait_time = 0.0
                
                self.max_wait_time_sec = max(self.max_wait_time_sec, wait_time)
                
                total_accumulated_wait += wait_time
                num_waiting_vehicles += 1
                if wait_time > current_max_wait:
                    current_max_wait = wait_time
                
                if edge_id not in edge_max_waits or wait_time > edge_max_waits[edge_id]:
                    edge_max_waits[edge_id] = wait_time
                
                if speed < 0.1:
                    standing_penalty += priority_mult
            else:
                # Vehicle has reached the intersection or an outgoing edge
                current_passed.add(vid)
        
        pass
        # Vehicles that arrived at their destination and were removed from simulation this step
        arrived_vehicles = set(traci.simulation.getArrivedIDList()) | self.arrived_during_transition
        
        newly_passed = (current_passed | arrived_vehicles) - self.passed_vehicles
        self.passed_vehicles.update(newly_passed)
        
        passed_reward = sum(self.priority_vehicles_seen.get(vid, 1.0) for vid in newly_passed)

        # Calculate Trip Quality
        trip_quality_sum = 0.0
        for vid in newly_passed:
            priority_mult = self.priority_vehicles_seen.get(vid, 1.0)
            if vid in present_ids:
                wait_time = float(traci.vehicle.getAccumulatedWaitingTime(vid))
            else:
                # Vehicle arrived and was removed from the simulation this step.
                wait_time = 0.0

            trip_quality_sum += priority_mult / (1.0 + wait_time)

        # Normalizing Trip Quality:
        norm_trip_quality = trip_quality_sum / (self.max_throughput_per_step * sim_steps_passed)

        # Normalization scaling factor based on max expected vehicles
        max_capacity = float(self.max_vehicles_in_lane * VEHICLE_SLOT_COUNT)
        if max_capacity <= 0:
            max_capacity = 1.0

        # Moving approach cars over all (standing + moving approach) cars
        moving_ratio = approaching_moving / max(1.0, total_standing + approaching_moving)
        weighted_moving_ratio = weighted_approaching_moving / max(1.0, total_standing + weighted_approaching_moving)

        # Penalize presence, penalize standing, reward passing
        norm_veh_flow_reward = (passed_reward - standing_penalty) / max_capacity
        
        # Calculate Wait metrics
        current_max_wait_norm = current_max_wait / self.max_wait_per_vehicle
        total_wait_norm = total_accumulated_wait / ((self.max_wait_per_vehicle/2) * VEHICLE_SLOT_COUNT)
        
        # Calculate fair queue imbalance (average of active lane leaders)
        if edge_max_waits:
            avg_queue_wait = sum(edge_max_waits.values()) / len(edge_max_waits)
            avg_queue_wait_norm = avg_queue_wait / self.max_wait_per_vehicle
        else:
            avg_queue_wait_norm = 0.0
            
        # Wait delta in raw seconds, per step (positive = improved)
        raw_wait_delta = (self.prev_total_accumulated_wait - total_accumulated_wait) / sim_steps_passed
        
        # Normalize the raw delta
        total_wait_delta = raw_wait_delta / max_capacity if max_capacity > 0 else 0.0
        total_wait_delta = raw_wait_delta / (sim_steps_passed * VEHICLE_SLOT_COUNT) if (sim_steps_passed * VEHICLE_SLOT_COUNT) > 0 else 0.0

        # Clip normalized delta strictly between -1.5 and 1.5
        total_wait_delta = max(-1.5, min(total_wait_delta, 1.5))

        # Update for next step with the RAW wait time in seconds
        self.prev_total_accumulated_wait = total_accumulated_wait

        # Clip metrics to 2.0 to prevent runaway imbalance values during catastrophic gridlocks
        current_max_wait_norm = min(current_max_wait_norm, 2.0)
        avg_queue_wait_norm = min(avg_queue_wait_norm, 2.0)

        imbalance = current_max_wait_norm - avg_queue_wait_norm


        # PEDESTRIANS (reward moving ratio)
        ped_ids = traci.person.getIDList()
        if ped_ids:
            speeds = np.array([traci.person.getSpeed(p) for p in ped_ids], dtype=np.float32)
            standing_peds = float(np.sum(speeds > 0.1))
            ped_reward = standing_peds / len(ped_ids)
        else:
            ped_reward = 0.0
            
        #print(f"Trip: {norm_trip_quality:.4f}\t\tWait delta: {total_wait_delta:.4f}\t\tMoving: {moving_ratio*0.1:.4f}\t\tPed: {ped_reward*0.1:.4f}")
        #print(f"Trip: {norm_trip_quality:.4f}\t\tWait delta: {total_wait_delta:.4f}\t\tMoving: {-imbalance*0.25:.4f}\t\tPed: {ped_reward*0.1:.4f}")
        #return (norm_veh_flow_reward - (imbalance * 0.25) + total_wait_delta + (ped_reward * 0.10)) * 0.2

        # GOLD -> return norm_trip_quality + total_wait_delta + (moving_ratio * 0.1) + (ped_reward * 0.1)
        # need to add:
        # normalization_factor = ((self.max_wait_per_vehicle/2) * VEHICLE_SLOT_COUNT)
        # total_wait_delta = raw_wait_delta / normalization_factor if normalization_factor > 0 else 0.0

        # return norm_trip_quality + (total_wait_delta*0.5) + (moving_ratio * 0.1) + (ped_reward * 0.1)

        # -----------------------
        # NEW ISOLATED REWARD
        # -----------------------
        lane_norm_counts = []
        for slot in self._vehicle_slots:
            if not slot.active:
                continue
                
            num_lanes = len(slot.sumo_lane_ids) if slot.sumo_lane_ids else 1
            if slot.max_capacity > 0:
                slot_max_vehicles = float(slot.max_capacity)
            else:
                slot_max_vehicles = self.max_vehicles_in_lane * num_lanes
                
            total_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in slot.sumo_lane_ids)
            lane_norm_counts.append(total_vehicles / slot_max_vehicles if slot_max_vehicles > 0 else 0.0)

        if lane_norm_counts:
            max_norm_veh = max(lane_norm_counts)
            avg_norm_veh = sum(lane_norm_counts) / len(lane_norm_counts)
            veh_count_imbalance = max_norm_veh - avg_norm_veh
        else:
            veh_count_imbalance = 0.0

        # return -veh_count_imbalance
        
        # -----------------------
        # NEW PRESSURE REWARD
        # -----------------------
        leaving_count = len(newly_passed)
        pressure = self.prev_arrived - leaving_count
        self.prev_arrived = self.current_arrived
        
        pressure_norm = pressure / max_capacity if max_capacity > 0 else 0.0
        
        # return -float(pressure_norm)

        # return norm_trip_quality * 0.5
        # return (total_wait_delta * 0.5)
        # return pressure_norm
        # Ablation reward switch (REWARD_MODE env); "full" is the default formula
        if _REWARD_MODE == "trip":      # trip-quality only
            return norm_trip_quality
        if _REWARD_MODE == "dwait":     # wait-delta only
            return total_wait_delta
        if _REWARD_MODE == "flow":      # flow-based reward
            return norm_veh_flow_reward + total_wait_delta + (moving_ratio * 0.1) + (ped_reward * 0.1)
        if _REWARD_MODE == "dwait_w1":  # higher wait-delta weight
            return norm_trip_quality + (total_wait_delta * 1.0) + (moving_ratio * 0.1) + (ped_reward * 0.1)
        return (norm_trip_quality + total_wait_delta + (moving_ratio * 0.1) + (ped_reward * 0.1))  # full (default)
        return norm_trip_quality + (total_wait_delta * 0.5) + (moving_ratio * 0.1) + (ped_reward * 0.1) - float(pressure_norm)

    def _compute_terminal_reward(self) -> float:
        """Terminal reward at episode end: combines average wait, worst wait, stranded vehicles, and speed bonus."""
        total_vehicles = max(len(self.vehicles_seen), 1)
        average_wait_norm = (self.total_cumulative_wait_time / total_vehicles) / self.max_wait_per_vehicle
        
        # Cap the average wait penalty to prevent massive swings during gridlock
        average_wait_norm = min(average_wait_norm, 2.0)
        
        stranded_norm = len(traci.vehicle.getIDList()) / (self.max_vehicles_in_lane * VEHICLE_SLOT_COUNT)
        
        # Reward proportional to the amount of simulation time saved (early finish)
        time_saved = max(0.0, EPISODE_DURATION - traci.simulation.getTime())
        time_saved_bonus = 10.0 * (time_saved / EPISODE_DURATION)

        # Clearance quality - continuous intersection clearing improvement
        clearance_quality = time_saved_bonus - stranded_norm

        # Individual driver satisfaction - prevent waiting (max_cumulative_wait_time is already capped during the step)
        driver_satisfaction = - self.max_cumulative_wait_time - average_wait_norm

        return clearance_quality + driver_satisfaction
    

    def _log_episode_summary(self, terminal_reward: float) -> None:
        total_vehicles = max(len(self.vehicles_seen), 1)
        average_wait_sec = self.total_cumulative_wait_time / total_vehicles
        level_name = (
            self.curriculum.levels[self.curriculum_level_index].name
            if self.curriculum else "fixed"
        )
        current_time = traci.simulation.getTime()

        print("EPISODE COMPLETED")
        print(f"  Curriculum level:          {level_name}")
        print(f"  Completion time:           {current_time:.0f}s / {EPISODE_DURATION}s")
        print(f"  Total cumulative wait:     {self.total_cumulative_wait_time:.0f}s")
        print(f"  Average wait per vehicle:  {average_wait_sec:.1f}s")
        print(f"  Max wait (normalised):     {self.max_cumulative_wait_time:.4f}")
        print(f"  Max wait (seconds):        {self.max_wait_time_sec:.0f}s")
        print(f"  Terminal reward:           {terminal_reward:.4f}")
        print("=" * 45)


def linear_schedule(initial_value: float):
    """Return a linear learning rate schedule that decays to 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def cushion_learning_rate(
    target_lr:        float,
    source_timesteps: int,
    target_timesteps: int,
) -> float:
    """Starting LR for continued training so the schedule still reaches target_lr at the combined run's end."""
    progress_remaining = 1.0 - source_timesteps / (source_timesteps + target_timesteps)
    if progress_remaining <= 0:
        return 1e-5
    return target_lr / progress_remaining


if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv

    # Prefer physical cores for SUMO; adjust to the host machine
    NUM_CORES = int(os.environ.get("NUM_CORES", "4"))

    # -- Curriculum definition -------------------------------------------
    _, _, bounds = load_intersection_config("configs/config_hornbach.yaml")
    sumocfg_path = bounds.get("sumocfg_path", "")
    
    # We now pass lists of dicts indicating yaml definitions instead of basic paths
    config_calibration = {"yaml": "config_hornbach.yaml", "sumocfg": "simulation/Hornbach-Calibration.sumocfg"}
    config_tutorial_free = {"yaml": "config_hornbach.yaml", "sumocfg": "simulation/Hornbach-Tutorial.sumocfg"}
    config_tutorial_restricted = {"yaml": "config_canonical-restricted.yaml", "sumocfg": "simulation/Hornbach-Tutorial.sumocfg"}
    config_easy = {"yaml": "config_hornbach.yaml", "sumocfg": "simulation/Hornbach-Easy.sumocfg"}
    config_medium = {"yaml": "config_hornbach.yaml", "sumocfg": "simulation/Hornbach-Medium.sumocfg"}
    config_hard = {"yaml": "config_hornbach.yaml", "sumocfg": "simulation/Hornbach-Hard.sumocfg"}
    config_hodoninska_tutorial = {"yaml": "config_hodoninska-train.yaml", "sumocfg": "simulation/Hodoninska-train-Tutorial.sumocfg"}
    config_hodoninska_easy = {"yaml": "config_hodoninska-train.yaml", "sumocfg": "simulation/Hodoninska-train-Easy.sumocfg"}
    config_hodoninska_medium = {"yaml": "config_hodoninska-train.yaml", "sumocfg": "simulation/Hodoninska-train-Medium.sumocfg"}
    config_hodoninska_hard = {"yaml": "config_hodoninska-train.yaml", "sumocfg": "simulation/Hodoninska-train-Hard.sumocfg"}
    config_aupark_tutorial = {"yaml": "config_aupark-train.yaml", "sumocfg": "simulation/Aupark-train-Tutorial.sumocfg"}
    config_aupark_easy = {"yaml": "config_aupark-train.yaml", "sumocfg": "simulation/Aupark-train-Easy.sumocfg"}
    config_aupark_medium = {"yaml": "config_aupark-train.yaml", "sumocfg": "simulation/Aupark-train-Medium.sumocfg"}
    config_aupark_hard = {"yaml": "config_aupark-train.yaml", "sumocfg": "simulation/Aupark-train-Hard.sumocfg"}
    config_canonical_tutorial = {"yaml": "config_canonical.yaml", "sumocfg": "simulation/Canonical-Tutorial.sumocfg"}
    config_canonical_easy = {"yaml": "config_canonical.yaml", "sumocfg": "simulation/Canonical-Easy.sumocfg"}
    config_canonical_medium = {"yaml": "config_canonical.yaml", "sumocfg": "simulation/Canonical-Medium.sumocfg"}
    config_canonical_hard = {"yaml": "config_canonical.yaml", "sumocfg": "simulation/Canonical-Hard.sumocfg"}
    config_aupark_prod_tutorial = {"yaml": "config_aupark-test.yaml", "sumocfg": "simulation/Aupark-Tutorial.sumocfg"}
    config_aupark_prod_easy = {"yaml": "config_aupark-test.yaml", "sumocfg": "simulation/Aupark-Easy.sumocfg"}
    config_aupark_prod_medium = {"yaml": "config_aupark-test.yaml", "sumocfg": "simulation/Aupark-Medium.sumocfg"}
    config_aupark_prod_hard = {"yaml": "config_aupark-test.yaml", "sumocfg": "simulation/Aupark-Hard.sumocfg"}
    config_galeria_prod_tutorial = {"yaml": "config_galeria-test.yaml", "sumocfg": "simulation/Galeria-Tutorial.sumocfg"}
    config_galeria_prod_easy = {"yaml": "config_galeria-test.yaml", "sumocfg": "simulation/Galeria-Easy.sumocfg"}
    config_galeria_prod_medium = {"yaml": "config_galeria-test.yaml", "sumocfg": "simulation/Galeria-Medium.sumocfg"}
    config_galeria_prod_hard = {"yaml": "config_galeria-test.yaml", "sumocfg": "simulation/Galeria-Hard.sumocfg"}


    """curriculum = CurriculumSchedule(levels=[
        CurriculumLevel(
            name="Tutorial",
            configs=[config_tutorial_free],
            step_threshold=300_000,
        ),
        CurriculumLevel(
            name="Tutorial restricted",
            configs=[config_canonical_tutorial, config_tutorial_restricted],
            step_threshold=400_000,
        ),
        CurriculumLevel(
            name="Level 1 — Easy",
            configs=[config_canonical_easy] * 3 + [config_canonical_tutorial] * 2 + [config_tutorial_restricted],
            step_threshold=600_000,
        ),
        CurriculumLevel(
            name="Level 2 — Medium",
            configs=[config_canonical_medium] * 3 + [config_canonical_easy, config_canonical_tutorial, config_tutorial_restricted, config_aupark_tutorial, config_hodoninska_tutorial],
            step_threshold=1_200_000,
        ),
        CurriculumLevel(
            name="Level 3 — Hard",
            configs=
                [config_canonical_hard] * 4 +
                [config_canonical_easy, config_canonical_tutorial] * 3 +
                [config_hodoninska_hard] * 2 +
                [config_hodoninska_easy, config_hodoninska_tutorial] +
                [config_aupark_hard] * 2 +
                [config_aupark_easy, config_aupark_tutorial],
            step_threshold = 2_000_000,
        ),
        CurriculumLevel(
            name="Level 4 — Stabilisation",
            configs=
            [config_aupark_prod_tutorial, config_aupark_prod_easy, config_aupark_prod_medium, config_aupark_prod_hard] +
            [config_galeria_prod_tutorial, config_galeria_prod_easy, config_galeria_prod_medium, config_galeria_prod_hard],
        ),
    ])"""
    if os.environ.get("ABLATE_NO_CURRICULUM") == "1":
        # Ablation: train directly on hard configs, no warm-up mixing
        curriculum = CurriculumSchedule(levels=[
            CurriculumLevel(
                name="No curriculum (hard only)",
                configs=[config_canonical_hard, config_hodoninska_hard,
                         config_aupark_prod_hard, config_galeria_prod_hard],
            ),
        ])
        print("[ablation] curriculum disabled: training on hard configs only")
    else:
        curriculum = CurriculumSchedule(levels=[
            CurriculumLevel(
                name="Level 3 — Hard",
                configs=
                [config_canonical_hard] * 3 +
                [config_canonical_easy, config_canonical_tutorial] * 2  +
                [config_hodoninska_hard] * 2 +
                [config_hodoninska_easy, config_hodoninska_tutorial] +
                [config_aupark_prod_hard] * 2 +
                [config_aupark_prod_easy, config_aupark_prod_tutorial] +
                [config_galeria_prod_hard] * 2 +
                [config_galeria_prod_easy, config_galeria_prod_tutorial],
            ),
        ])

    # -- Controller & environment ----------------------------------------
    def make_env(rank: int, curriculum_schedule: CurriculumSchedule):
        def _init():
            # Initialise isolated controller for this specific process
            first_yaml = curriculum_schedule.levels[0].configs[0]["yaml"] if getattr(curriculum_schedule, 'levels', None) else "config_hornbach.yaml"
            c, rs, b = load_intersection_config(first_yaml)
            return SumoIntersectionEnv(
                controller  = c,
                bounds      = b,
                curriculum  = curriculum_schedule,
                route_slots = rs,
            )
        return _init

    env = SubprocVecEnv([make_env(i, curriculum) for i in range(NUM_CORES)])

    # -- Configurable ALGO -----------------------------------------------
    ALGO = os.environ.get("ALGO", "PPO").upper()
    algo_class = SAC if ALGO == "SAC" else PPO

    # -- Callbacks -------------------------------------------------------
    # Per-seed tag so concurrent seeds don't overwrite each other's checkpoints.
    _seed_tag = f"_seed{os.environ['SEED']}" if os.environ.get("SEED") else ""
    curriculum_callback = CurriculumCallback(curriculum_schedule=curriculum, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = f"./logs{_seed_tag}/",
        name_prefix = f"sumo_{ALGO.lower()}{_seed_tag}",
    )

    # -- Model -----------------------------------------------------------
    reward_strategy = os.environ.get("REWARD_MODE", "full")
    mission = "ATLAS5"
    mission += ("_noCurriculum" if os.environ.get("ABLATE_NO_CURRICULUM") == "1" else "") + ("_noLead" if os.environ.get("ABLATE_NO_LEAD") == "1" else "")
    mission += (f"_seed{os.environ['SEED']}" if os.environ.get("SEED") else "")  # keep multi-seed models distinct
    model_name = None
    total_timesteps = int(os.environ.get("TRAIN_STEPS", "3000000"))
    target_learning_rate = 0.00035
    expert_buffer_path = "simulation/expert_human_buffer_config_hornbach.pkl"

    algo_kwargs = {}
    if ALGO == "SAC":
        algo_kwargs["ent_coef"] = 'auto'

    if model_name is not None:
        model = algo_class.load(model_name, env=env, learning_rate=target_learning_rate, **algo_kwargs)
        reset_num_timesteps = False
    else:
        model = algo_class(
            "MlpPolicy",
            env,
            verbose         = 1,
            gamma           = 0.99,
            learning_rate   = target_learning_rate,
            device          = os.environ.get("DEVICE", "auto"),
            tensorboard_log = f"./{ALGO.lower()}_sumo_tensorboard/",
            seed            = (int(os.environ["SEED"]) if os.environ.get("SEED") else None),
            **algo_kwargs
        )
        reset_num_timesteps = True

    # Load expert buffer if present (SB3 appends ".pkl" to the path); else train SAC unseeded
    if expert_buffer_path is not None and ALGO == "SAC":
        _disk = expert_buffer_path if expert_buffer_path.endswith(".pkl") else expert_buffer_path + ".pkl"
        _stem = _disk[:-4]
        if os.path.exists(_disk):
            print(f"Loading expert buffer from {_disk}...")
            model.load_replay_buffer(_stem)
            # Set learning_starts so that non-expert data makes up ~10% of the initial experiences
            buffer_size = model.replay_buffer.size()
            model.learning_starts = int(buffer_size / 9)
            print(f"Expert buffer loaded with {buffer_size} transitions. Learning starts at step {model.learning_starts}.")
        else:
            print(f"[info] Expert buffer not found ({_disk}); training SAC unseeded.")

    # Infer the next flight number (matches files like ATLAS4_SAC_Flight1 or ATLAS4_RewardName_Flight2)
    existing_flights = [
        int(re.search(r"Flight(\d+)", f).group(1))
        for f in os.listdir(".")
        if f.startswith(mission+"_") and re.search(r"Flight(\d+)", f)
    ]
    flight_number = max(existing_flights) + 1 if existing_flights else 1

    print(f"Starting training (Algo: {ALGO} | Flight: {flight_number} | Reward: {reward_strategy})...")
    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [curriculum_callback, checkpoint_callback],
        reset_num_timesteps = reset_num_timesteps,
        tb_log_name=f"[{mission}_{ALGO}]-({reward_strategy})-{flight_number}",
    )

    final_model_path = f"{mission}_{ALGO}_{reward_strategy}_Flight{flight_number}"
    model.save(final_model_path)
    print(f"Training complete. Model saved as {final_model_path}.")
    print(f"Training complete. Model saved as {mission}_{reward_strategy}_Flight{flight_number}.")
