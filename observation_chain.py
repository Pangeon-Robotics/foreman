"""FEP Observation Chain: state definitions and mock layer implementations.

Defines the inter-layer state dataclasses (LowState, KinematicState,
DynamicState, BehavioralState) and mock layer implementations for testing.
"""

import numpy as np
from dataclasses import dataclass


# ============================================================================
# LAYER STATE DEFINITIONS (Semantic Names)
# ============================================================================

@dataclass
class LowState:
    """Layer 2 -> Layer 3: Raw actuator/sensor data (SDK-defined)."""
    motor_positions: np.ndarray      # (12,) joint angles in radians
    motor_velocities: np.ndarray     # (12,) joint velocities in rad/s
    motor_torques: np.ndarray        # (12,) estimated torques in N-m
    imu_quaternion: np.ndarray       # (4,) orientation (w, x, y, z)
    imu_gyroscope: np.ndarray        # (3,) angular velocity in rad/s
    imu_accelerometer: np.ndarray    # (3,) linear acceleration in m/s^2
    foot_forces: np.ndarray          # (4, 3) contact forces in sensor frame
    timestamp: float


@dataclass
class KinematicState:
    """Layer 3 -> Layer 4: Body kinematics in world frame.

    Layer 3 "explains away" raw sensor data by computing:
    - Forward kinematics: joints -> foot positions
    - Frame transforms: sensor frame -> world frame
    - Contact detection: force threshold -> binary state

    What Layer 3 cannot explain (errors) are passed to Layer 4.
    """
    foot_positions_world: np.ndarray   # (4, 3) meters, world frame
    foot_velocities_world: np.ndarray  # (4, 3) m/s, world frame
    foot_contact_states: np.ndarray    # (4,) boolean contact flags
    foot_contact_forces: np.ndarray    # (4, 3) Newtons, world frame
    base_orientation: np.ndarray       # (4,) quaternion
    base_angular_velocity: np.ndarray  # (3,) rad/s
    timestamp: float

    # Layer 3's "unexplainable" component (kinematic error)
    kinematic_error: float = 0.0       # Residual FK error magnitude


@dataclass
class DynamicState:
    """Layer 4 -> Layer 5: Trajectory predictions and dynamics errors.

    Layer 4 "explains away" kinematic observations by predicting:
    - Physics-based trajectory (HNN or analytical)
    - Expected contact forces from terrain model
    - Velocity evolution from dynamics

    What Layer 4 cannot explain (prediction errors) are passed to Layer 5.
    """
    # What Layer 4 predicted
    predicted_foot_positions: np.ndarray   # (4, 3) meters
    predicted_foot_velocities: np.ndarray  # (4, 3) m/s
    predicted_contact_forces: np.ndarray   # (4, 3) Newtons

    # What Layer 3 actually observed
    actual_kinematics: KinematicState

    # Prediction errors (surprise!)
    position_error: np.ndarray          # (4, 3) meters
    velocity_error: np.ndarray          # (4, 3) m/s
    force_error: np.ndarray             # (4, 3) Newtons

    # Inferred terrain properties (from errors)
    terrain_stiffness: float            # Pa, inferred from force errors
    terrain_friction: float             # Coefficient
    terrain_surprise: float             # std(errors) over recent window

    # Gait state
    gait_phase: float                   # [0, 1) within cycle
    timestamp: float


@dataclass
class BehavioralState:
    """Layer 5 -> Layer 6: High-level gait/terrain model and strategic state.

    Layer 5 "explains away" dynamic errors by modeling:
    - Terrain type (flat, stairs, rough) from dynamics errors
    - Gait stability/success from prediction accuracy
    - Expected velocity outcomes from gait model

    What Layer 5 cannot explain (strategic failures) are passed to Layer 6.
    """
    # Gait state
    current_gait: str                   # 'walk', 'trot', 'bound', 'stand'
    gait_stability: float               # [0, 1] confidence

    # Terrain belief
    terrain_type: str                   # 'flat', 'stairs', 'slope', 'rough'
    terrain_confidence: float           # [0, 1] belief certainty

    # Forward predictions
    expected_velocity: np.ndarray       # (3,) m/s
    velocity_variance: np.ndarray       # (3,) uncertainty

    # Free energy components
    pragmatic_value: float              # Goal achievement
    epistemic_value: float              # Information gain

    timestamp: float


# ============================================================================
# MOCK LAYERS (Test Implementation)
# ============================================================================

class Layer2_Firmware:
    """Mock firmware that generates sensor data."""

    def __init__(self):
        self.t = 0.0

    def step(self, dt: float = 0.01) -> LowState:
        """Generate mock sensor data."""
        self.t += dt

        # Simulate walking: sinusoidal joint motion
        phase = 2 * np.pi * self.t
        q = 0.3 * np.sin(phase) * np.ones(12)
        dq = 0.3 * 2 * np.pi * np.cos(phase) * np.ones(12)
        tau = 50 * np.sin(phase) * np.ones(12)

        # Simulate foot contacts (alternating)
        forces = np.zeros((4, 3))
        if np.sin(phase) > 0:  # FR, RL in contact
            forces[0, 2] = 200  # FR vertical force
            forces[3, 2] = 200  # RL vertical force
        else:  # FL, RR in contact
            forces[1, 2] = 200
            forces[2, 2] = 200

        return LowState(
            motor_positions=q,
            motor_velocities=dq,
            motor_torques=tau,
            imu_quaternion=np.array([1, 0, 0, 0]),
            imu_gyroscope=np.zeros(3),
            imu_accelerometer=np.array([0, 0, 9.81]),
            foot_forces=forces,
            timestamp=self.t
        )


class Layer3_Kinematics:
    """Mock IK layer: transforms sensor data -> world frame kinematics."""

    def __init__(self):
        self.prev_positions = None

    def process(self, lowstate: LowState) -> KinematicState:
        """Transform LowState -> KinematicState (upward observation)."""

        # Mock FK: joint angles -> foot positions
        q = lowstate.motor_positions
        foot_pos = np.array([
            [0.3, -0.15, -0.3 + 0.1 * np.sin(q[0])],  # FR
            [0.3,  0.15, -0.3 + 0.1 * np.sin(q[3])],  # FL
            [-0.3, -0.15, -0.3 + 0.1 * np.sin(q[6])], # RR
            [-0.3,  0.15, -0.3 + 0.1 * np.sin(q[9])], # RL
        ])

        # Mock velocity: finite difference
        dt = 0.01
        if self.prev_positions is not None:
            foot_vel = (foot_pos - self.prev_positions) / dt
        else:
            foot_vel = np.zeros((4, 3))
        self.prev_positions = foot_pos.copy()

        # Contact detection: force threshold
        contact_threshold = 50.0  # Newtons
        contacts = np.linalg.norm(lowstate.foot_forces, axis=1) > contact_threshold

        # Frame transform: sensor -> world (simplified, no rotation)
        forces_world = lowstate.foot_forces.copy()

        # Kinematic error: residual FK mismatch (simulated)
        kinematic_error = 0.01 * np.random.randn()

        return KinematicState(
            foot_positions_world=foot_pos,
            foot_velocities_world=foot_vel,
            foot_contact_states=contacts,
            foot_contact_forces=forces_world,
            base_orientation=lowstate.imu_quaternion,
            base_angular_velocity=lowstate.imu_gyroscope,
            timestamp=lowstate.timestamp,
            kinematic_error=kinematic_error
        )


class Layer4_Dynamics:
    """Mock dynamics layer: predicts trajectories, computes errors."""

    def __init__(self):
        self.terrain_stiffness = 1e6  # Initial guess: concrete
        self.error_history = []

    def process(self, kin_state: KinematicState, gait_phase: float) -> DynamicState:
        """Predict dynamics and compute errors (upward observation)."""

        # Mock prediction: simple sinusoidal trajectory
        phase = 2 * np.pi * gait_phase
        predicted_pos = kin_state.foot_positions_world + 0.05 * np.sin(phase)
        predicted_vel = 0.05 * 2 * np.pi * np.cos(phase) * np.ones((4, 3))

        # Predict contact forces from terrain model
        predicted_forces = np.zeros((4, 3))
        for i, contact in enumerate(kin_state.foot_contact_states):
            if contact:
                # F = k * penetration (simplified)
                penetration = max(0, -kin_state.foot_positions_world[i, 2])
                predicted_forces[i, 2] = self.terrain_stiffness * penetration

        # Compute prediction errors (SURPRISE!)
        pos_error = kin_state.foot_positions_world - predicted_pos
        vel_error = kin_state.foot_velocities_world - predicted_vel
        force_error = kin_state.foot_contact_forces - predicted_forces

        # Update terrain model from errors
        force_error_magnitude = np.linalg.norm(force_error)
        self.error_history.append(force_error_magnitude)
        if len(self.error_history) > 100:
            self.error_history.pop(0)

        # Infer terrain stiffness from force errors
        if force_error_magnitude > 100:
            self.terrain_stiffness *= 0.95  # Softer than expected
        elif force_error_magnitude < 10:
            self.terrain_stiffness *= 1.05  # Harder than expected

        terrain_surprise = np.std(self.error_history) if self.error_history else 0.0

        return DynamicState(
            predicted_foot_positions=predicted_pos,
            predicted_foot_velocities=predicted_vel,
            predicted_contact_forces=predicted_forces,
            actual_kinematics=kin_state,
            position_error=pos_error,
            velocity_error=vel_error,
            force_error=force_error,
            terrain_stiffness=self.terrain_stiffness,
            terrain_friction=0.8,  # Mock
            terrain_surprise=terrain_surprise,
            gait_phase=gait_phase,
            timestamp=kin_state.timestamp
        )


class Layer5_Behavior:
    """Mock behavior layer: gait selection via FEP."""

    def __init__(self):
        self.gait_models = {
            'walk': {'success_rate': 0.95, 'speed': 0.5},
            'trot': {'success_rate': 0.80, 'speed': 1.0},
            'bound': {'success_rate': 0.60, 'speed': 1.5},
        }
        self.current_gait = 'trot'

    def process(self, dyn_state: DynamicState, velocity_cmd: float) -> BehavioralState:
        """Select gait via FEP (minimize expected free energy)."""

        # Classify terrain from dynamics errors
        if dyn_state.terrain_surprise > 50:
            terrain_type = 'rough'
            terrain_conf = 0.8
        elif dyn_state.terrain_surprise > 20:
            terrain_type = 'stairs'
            terrain_conf = 0.7
        else:
            terrain_type = 'flat'
            terrain_conf = 0.9

        # FEP gait selection: minimize expected free energy
        efe = {}
        for gait, model in self.gait_models.items():
            # Pragmatic value: P(success | gait, terrain)
            pragmatic = model['success_rate']
            if terrain_type == 'rough':
                pragmatic *= 0.5 if gait == 'trot' else 1.0

            # Epistemic value: information gain (exploration bonus)
            epistemic = 0.1 if gait != self.current_gait else 0.0

            # Expected Free Energy = -pragmatic + lambda*epistemic
            efe[gait] = -pragmatic + 0.1 * epistemic

        # Select gait with minimum EFE
        selected_gait = min(efe, key=efe.get)
        self.current_gait = selected_gait

        # Predict velocity outcome
        expected_vel = np.array([self.gait_models[selected_gait]['speed'], 0, 0])
        vel_variance = np.array([0.1, 0.05, 0.05])

        return BehavioralState(
            current_gait=selected_gait,
            gait_stability=self.gait_models[selected_gait]['success_rate'],
            terrain_type=terrain_type,
            terrain_confidence=terrain_conf,
            expected_velocity=expected_vel,
            velocity_variance=vel_variance,
            pragmatic_value=-efe[selected_gait],
            epistemic_value=epistemic,
            timestamp=dyn_state.timestamp
        )
