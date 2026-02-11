# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

This is a **multi-layer robotics control stack** for Unitree SDK2 robots (B2, Go2, H1, G1). The codebase implements an 8-layer onion architecture where each layer translates from one control language to another, following the OSI network model pattern.

**Repository structure**: Mono-repo containing separate implementations of Layers 1-5, plus a `philosophy/` directory with canonical architecture documentation and engineering principles compiled from 40+ postmortems.

## Mono-Repo Structure and Sovereignty

This directory contains **multiple sovereign layer implementations** that work together but remain separate:

- **Layers** (layers_1_2/, layer_3/, layer_4/, layer_5/): Each is a sovereign implementation with its own VERSION, CLAUDE.md, and API contract
- **Philosophy** (philosophy/): Source of truth for architecture, principles, and workflows
- **Training** (training/): Separate Git repository for offline ML training (HNN, RL)
- **Improvements** (improvements/): Architectural planning and research proposals
- **Assets** (Assets/): Shared MuJoCo robot models (layers_1_2/unitree_robots/ symlinks here)

**Key principle**: Each layer has **agency, scope, and responsibility**. Layers may read other layers for reference but **NEVER edit files belonging to another layer**. To request changes, use `philosophy/workflows/fix-request.json`.

See `philosophy/scope.md` for information economy principles and `philosophy/boundaries.md` for boundary enforcement.

## The 8-Layer Architecture

| Layer | Directory | Status | Purpose |
|-------|-----------|--------|---------|
| **Layer 1** | `layers_1_2/` | ✅ Complete | Motor Drivers / Physics (MuJoCo simulation) |
| **Layer 2** | `layers_1_2/` | ✅ Complete | Firmware PD Control (`firmware_sim.py`) |
| **Layer 3** | `layer_3/` | ✅ Complete | Inverse Kinematics (Cartesian → joint angles) |
| **Layer 4** | `layer_4/` | ✅ Complete | Cartesian Positions (gait params → foot positions) |
| **Layer 5** | `layer_5/` | 🚧 In Progress | Locomotion (motion commands → gait parameters) |
| **Layer 6** | Not implemented | Waypoints and Tasks |
| **Layer 7** | Not implemented | Mission Planning |
| **Layer 8** | Not implemented | Application UI |

**Key insight**: Layers 1-4 are **instant** (per-timestep translation, no memory). Layers 5-8 are **sequences** (maintain state, plan over time). The DDS boundary sits between Layers 2-3.

## Quick Start

```bash
# Activate Python environment
source env/bin/activate

# Run the full stack (two terminals)
# Terminal 1: Start firmware (Layers 1-2)
cd layers_1_2 && python firmware_sim.py --robot b2

# Terminal 2: Run a controller (Layer 3)
cd layer_3 && python controller.py

# Or run tests for a specific layer
cd layer_3 && make test
```

## Repository Structure

```
/Users/graham/code/robotics/
├── philosophy/          # Source of truth: architecture, principles, workflows
│   ├── architecture.md  # Canonical 8-layer model
│   ├── scope.md         # Need-to-know, vocabulary compression, agency
│   ├── boundaries.md    # Layer sovereignty and violation types
│   └── workflows/       # Cross-repo workflows (fix-request, compliance)
│
├── layers_1_2/          # Layers 1-2 (sovereign implementation)
│   ├── CLAUDE.md        # Layer 1-2 specific guidance
│   ├── VERSION          # v0.1.11
│   ├── firmware_sim.py  # Virtual firmware + MuJoCo
│   ├── dds.py           # DDS initialization (shared)
│   └── unitree_robots/ → ../Assets/unitree_robots/  (symlink)
│
├── layer_3/             # Layer 3: IK (sovereign implementation)
│   ├── CLAUDE.md        # Layer 3 specific guidance
│   ├── VERSION          # v2.1.0
│   └── controller.py    # 100 Hz control loop
│
├── layer_4/             # Layer 4: Cartesian (sovereign implementation)
│   ├── CLAUDE.md        # Layer 4 specific guidance
│   ├── VERSION          # v0.13.0
│   └── generator.py     # Gait → positions
│
├── layer_5/             # Layer 5: Locomotion (sovereign, in progress)
│   ├── CLAUDE.md        # Layer 5 specific guidance
│   └── VERSION          # Under development
│
├── training/            # Separate Git repo: offline ML training
│   ├── .git/            # Own Git history
│   ├── CLAUDE.md        # Training-specific guidance
│   ├── hnn/             # Hamiltonian Neural Networks
│   └── rl/              # RL policies (PPO)
│
├── improvements/        # Architectural planning (separate Git repo)
│   ├── .git/            # Own Git history
│   ├── IMPLEMENTATION_PLAN.md  # Step-by-step HNN/RL plan
│   └── ARCHITECTURE_JOURNEY.md # Design decisions record
│
├── Assets/              # Shared robot models (separate Git repo)
│   ├── .git/            # Own Git history
│   ├── robots/          # XML definitions (b2.xml, g1.xml, etc.)
│   ├── meshes/          # 3D models by robot
│   └── scenes/          # Reusable environments
│
├── test_observation_chain.py  # Integration test for FEP architecture
└── env/                 # Python virtual environment
```

## Common Commands

### Firmware (Layers 1-2)
```bash
cd layers_1_2
make install           # Install dependencies
make test              # Run all tests
make run               # Start firmware with viewer
python firmware_sim.py --robot b2 --headless  # Headless mode
```

### Layer 3 (IK)
```bash
cd layer_3
make install           # Install dependencies
make test              # Run all tests
make run               # Run controller (firmware must be running)
python demos/stand.py  # Run stand demo
```

### Layer 4 (Cartesian)
```bash
cd layer_4
make install           # Install dependencies
make test              # Run all tests
```

### Layer 5 (Locomotion)
```bash
cd layer_5
make install           # Install dependencies
make test              # Run all tests (when implemented)
```

## Supporting Repositories

### Training (Offline ML)

The `training/` directory is a **separate Git repository** for offline machine learning:

```bash
cd training
make install              # Install training dependencies
python hnn/train.py       # Train HNN dynamics model
python rl/train_ppo.py    # Train RL locomotion policy
```

**Key principle**: Training happens **offline, outside the layer stack**. Layers consume frozen models via GitHub Releases, never train online. This maintains layer sovereignty—Layer 4 receives an HNN model artifact, Layer 5 receives an RL policy artifact.

**Models distributed via GitHub Releases** (not checked into layer repos):
- HNN models → layer_4 for terrain estimation
- RL policies → layer_5 for gait selection

See `training/CLAUDE.md` and `training/README.md` for training workflows.

### Improvements (Architectural Planning)

The `improvements/` directory contains **cross-layer architectural decisions**:

- **IMPLEMENTATION_PLAN.md**: Step-by-step plan for HNN/RL features
- **ARCHITECTURE_JOURNEY.md**: Historical record of design insights
- **LEARNING_ROADMAP.md**: Technical details for training

**Scope**: Documents proposals that span multiple layers. Individual layer changes go in `layer_X/plans/`.

See `improvements/README.md` for navigation.

### Assets (Robot Models)

The `Assets/` directory is the **central repository** for MuJoCo models:

- `Assets/robots/` — Robot definitions (b2.xml, go2.xml, h1.xml, etc.)
- `Assets/scenes/` — Reusable environments (flat.xml, terrain.xml)
- `Assets/meshes/` — 3D models organized by robot
- `Assets/unitree_robots/` — Full MuJoCo models for simulation

**Note**: `layers_1_2/unitree_robots/` is a **symlink** to `Assets/unitree_robots/`.

Download from: https://huggingface.co/unitree-robotics/unitree_robots

See `Assets/README.md` for usage patterns.

## Critical Principles

### 1. Layered Abstraction
- **Never use `if simulation:` branches**. Controller code must be identical whether talking to real robot or simulation.
- Each layer only uses sensors/data available on real hardware (no `data.qpos[0:3]` GPS coordinates, no `data.xpos` mocap)
- The Virtual MCU pattern: simulation emulates firmware PD control exactly

### 2. Layer Sovereignty
- Each layer is **sovereign** — you may read other layers for reference but **NEVER edit files belonging to another layer**
- To request changes in a lower layer, file a GitHub issue using `philosophy/workflows/fix-request.json`
- Each layer has its own CLAUDE.md, ARCHITECTURE.md, README.md, and VERSION file

### 3. Joint vs Actuator Order
**CRITICAL**: MuJoCo and DDS use different orderings:
- **MuJoCo body tree**: FL, FR, RL, RR
- **DDS actuators**: FR, FL, RR, RL
- **Permutation**: `[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]` (self-inverse)
- **Forgetting this causes immediate robot collapse**

### 4. DDS Communication
- **Never call `ChannelFactoryInitialize()`** — causes buffer overflow in CycloneDDS 0.10.4
- Always use `dds_init(domain_id=1, interface="lo")` from `dds.py`
- **Domain IDs**: 1 = simulation, 0 = real robot
- **CRC required**: Call `stamp_cmd(cmd)` before every `pub.Write(cmd)`

### 5. Hardware Abstraction
**Allowed** (available on real robot):
- Joint encoders: `motor_state[i].q`, `.dq`, `.tau_est`
- IMU: `imu_state.quaternion`, `.gyroscope`, `.accelerometer`
- Foot contact forces

**Forbidden** (simulation-only):
- `data.qpos[0:3]` — GPS/mocap global position
- `data.xpos` — mocap body positions
- Contact geometry details
- Anything requiring `import mujoco` in Layer 3+

### 6. Scope and Need-to-Know

From `philosophy/scope.md`:

- **Each layer receives only the information required for its task**. Unnecessary context creates wrong associations and invites scope creep.
- **Minimize vocabulary while keeping it sufficiently rich**. Each layer has its own vocabulary (Layer 3: positions/angles, Layer 5: gaits/velocities). Using words from another layer signals boundary violation.
- **The compression test**: Can you describe your layer's job in one sentence using only its vocabulary? If not, you're doing too much.

**Applied to agents**: When Claude operates in a specific layer's directory, it should:
1. Read that layer's CLAUDE.md first (establishes scope and boundaries)
2. Use only that layer's vocabulary and API
3. Never edit files in other layers (file fix-requests instead)
4. Focus on capabilities (what the layer can DO) not tools (what it HAS)

## Documentation

### Start Here
1. **[philosophy/start_here.md](philosophy/start_here.md)** — Onboarding guide (read first, ~45 min)
2. **[philosophy/architecture.md](philosophy/architecture.md)** — Canonical 8-layer reference
3. **Layer-specific CLAUDE.md** — Development guidelines for each layer

### Key Documents
- **[philosophy/scope.md](philosophy/scope.md)** — Need-to-know, vocabulary compression, capabilities vs tools
- **[philosophy/boundaries.md](philosophy/boundaries.md)** — Layer sovereignty, 3 violation types, 4 protection mechanisms
- **[philosophy/firmware_contract.md](philosophy/firmware_contract.md)** — Contract between firmware (Layer 2) and controllers (Layer 3+)
- **[philosophy/dds_boundary.md](philosophy/dds_boundary.md)** — DDS message structures, joint ordering, gain tables
- **[philosophy/control.md](philosophy/control.md)** — Sim2real fundamentals, PD control, frequency hierarchy
- **[philosophy/engineering.md](philosophy/engineering.md)** — Code structure: modularity, short files, no duplication

### Workflows
When the user says "P <workflow>", execute the corresponding workflow from `philosophy/workflows/`:
- **P fix-request** — File a GitHub issue in Layer N-1's repo
- **P resolve-issues** — Check for open issues, fix, test, close
- **P compliance** — Self-check repo against philosophy docs

## Working Across Layers

When Claude operates in this repository:

### If in a specific layer directory (layer_3/, layer_4/, etc.):
1. **Read that layer's CLAUDE.md first** — it defines scope, boundaries, and vocabulary
2. **Respect sovereignty** — never edit files in other layers
3. **Use fix-request workflow** — if Layer N needs changes in Layer N-1, file `P fix-request`
4. **Stay in vocabulary** — describe work using only that layer's concepts

### If in root directory or cross-layer work:
1. **Read philosophy/scope.md** — understand need-to-know and vocabulary compression
2. **Identify which layer owns the work** — don't mix layer concerns
3. **Route to appropriate layer** — each layer has its own agent context
4. **Use workflows** — `P compliance`, `P fix-request`, `P resolve-issues`

### Information flow direction:
- **Commands flow down** (Layer 5 → 4 → 3 → 2 → 1)
- **Observations flow up** (Layer 1 → 2 → 3 → 4 → 5)
- **Sovereignty flows nowhere** — each layer is independent

See `philosophy/architecture.md` for the canonical layer definitions.

## Development Workflow

### Two-Terminal Testing
```bash
# Terminal 1: Firmware
cd layers_1_2 && python firmware_sim.py --robot b2

# Terminal 2: Controller
cd layer_3 && python controller.py
```

### Automated Testing
```bash
# Each layer uses FirmwareLauncher for automatic firmware lifecycle
cd layer_3 && pytest tests/test_controller.py -v
```

### Cross-Layer Integration Tests

**test_observation_chain.py** — Root-level test validating the FEP Observation Chain:

```bash
python test_observation_chain.py
```

Tests:
- Strict N → N-1 layering (no layer skipping)
- Semantic naming (KinematicState, DynamicState, BehavioralState)
- FEP "explain away" pattern (error propagation upward)
- Latency constraints across layers

This is a **cross-layer integration test** — individual layers have their own unit tests in their own directories.

### Manual Exploration
```bash
# Print robot model info
cd layers_1_2 && python firmware_sim.py --robot b2 --info

# Run with viewer for debugging
python firmware_sim.py --robot b2  # Default: realtime with viewer, camera tracking

# Kill zombie processes if tests fail
pkill -9 -f "firmware_sim.py"
```

## Dependencies

### Core Requirements
- Python 3.12 (Ubuntu 24.04 default)
- MuJoCo 3.3.0+
- CycloneDDS 0.10.4 (**critical version**, 0.10.5+ has buffer overflow)
- NumPy 2.4.1+

### SDK (Vendored)
```bash
# Install Unitree SDK2 Python bindings (not on PyPI)
pip install -e unitree_sdk2_python/
```

### Robot Models
- Located in `Assets/unitree_robots/` (symlinked from `layers_1_2/unitree_robots/`)
- Download from: https://huggingface.co/unitree-robotics/unitree_robots
- Supported: b2, go2, h1, g1, b2w, go2w

### Environment Setup
```bash
# Create and activate virtual environment
python3.12 -m venv env
source env/bin/activate

# Install per-layer dependencies
cd layers_1_2 && make install
cd ../layer_3 && make install
cd ../layer_4 && make install
cd ../layer_5 && make install
```

## Architecture Diagrams

### Hardware Mapping
```
Real Robot                          Simulation
─────────────────────────          ─────────────────────────
Jetson Orin (Docker)               Dev machine
├─ Layer 5: Locomotion             ├─ Layer 5: Locomotion
├─ Layer 4: Cartesian              ├─ Layer 4: Cartesian
└─ Layer 3: IK                     └─ Layer 3: IK
     ↕ DDS (internal network)           ↕ DDS (loopback)
Intel CPU                          firmware_sim.py
├─ Layer 2: Firmware PD            ├─ Layer 2: PD control loop
└─ Layer 1: Motor drivers          └─ MuJoCo (replaces reality)
     ↕
Physical world (reality)
```

### Control Flow
```
Layer 5: Locomotion (motion commands)
    ↓
Layer 4: Cartesian (gait params → foot positions)
    ↓
Layer 3: IK (Cartesian → joint angles)
    ↓
DDS: rt/lowcmd, rt/lowstate
    ↓
Layer 2: Firmware (joint angles → torques via PD)
    ↓
Layer 1: Motors/Physics (torques → state)
```

## Troubleshooting

### DDS Initialization Errors
**Symptom**: "Failed to find a free participant index for domain 1"
**Cause**: Zombie firmware_sim processes from interrupted runs
**Fix**: `pkill -9 -f "firmware_sim.py"`

### Robot Collapses Immediately
**Cause 1**: Joint order confusion (MuJoCo vs DDS)
**Fix**: Use `JOINT_TO_ACTUATOR` permutation at boundaries

**Cause 2**: Unsigned commands (missing CRC)
**Fix**: Call `stamp_cmd(cmd)` before `pub.Write(cmd)`

**Cause 3**: Test gains too high (kp=2500 causes oscillation in sim)
**Fix**: Use test gains (kp=500, kd=25) for simulation

### Tests Fail to Launch Firmware
**Cause**: Multiple test files run in same pytest session
**Fix**: Use separate pytest invocations (see `layers_1_2/Makefile`)

## Versioning

Each layer maintains its own semantic version:
- **MAJOR**: Breaking changes to DDS interface or API
- **MINOR**: New features, new topics, API additions
- **PATCH**: Bug fixes, tests, documentation

Current versions (as of 2026-02-10):
- layers_1_2: v0.1.11
- layer_3: v2.1.0
- layer_4: v0.13.0
- layer_5: Under development

## Engineering Principles

From `philosophy/engineering.md`:
1. **Never duplicate** — Each concept lives in exactly one file
2. **Modular design** — Each file has a one-sentence purpose
3. **Short files** — Under 200 lines ideal, split if over 400
4. **Explicit code** — Named constants with units, visible imports
5. **Composition over inheritance** — Use protocols, not class hierarchies

## Additional Resources

- **[philosophy/gotchas.md](philosophy/gotchas.md)** — Lessons from 40+ postmortems
- **[philosophy/training.md](philosophy/training.md)** — Dense reward shaping, RL methodology
- **[philosophy/deployment.md](philosophy/deployment.md)** — Two-tier simulation strategy
- Layer-specific `ARCHITECTURE.md` files for implementation details
- Layer-specific `docs/api.md` files for public interfaces

---

**Last updated**: 2026-02-11
**Architecture version**: 8-layer onion model
**Applies to**: Unitree SDK2 robots (B2, Go2, G1, H1)
