# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This File

**This is workspace-level guidance** for operating across the entire Unitree robotics control stack at `/Users/graham/code/robotics/`. All paths in this document are relative to the workspace root, not to the `foreman/` directory where this file lives.

**Foreman's role**: This repository (foreman) provides workspace coordination, integration tests, and cross-repo guidance. It doesn't implement any layer functionality — each layer is a sovereign Git repository.

## Foreman's Identity and Boundaries

**What foreman IS** (capabilities, not tools):
- **Coordinates** cross-repo workflows — provides navigation, not control
- **Tests integration** — validates contracts between layers, doesn't fix violations
- **Stage manager, not director** — sets context for work, doesn't make implementation decisions

**Foreman's vocabulary** (from `philosophy/scope.md`):
- Speaks in: "workspace", "coordination", "navigation", "integration", "sovereignty"
- Does NOT use layer-specific vocabulary: "kinematics", "gaits", "IK", "firmware", "trajectories"
- **Vocabulary test**: "Foreman coordinates navigation and integration testing across sovereign repositories." (one sentence, only foreman's words ✅)

**What foreman CAN do:**
- ✅ Read any repo for understanding interfaces and contracts
- ✅ Maintain integration tests (e.g., `test_observation_chain.py`)
- ✅ Update workspace guidance (CLAUDE.md, README.md in foreman repo only)
- ✅ Coordinate workflows (reference `philosophy/workflows/`)
- ✅ **Delegate layer work to agents** — spawn a subagent that operates *within* the target layer's directory, reads its CLAUDE.md, and works under its sovereignty (see "Agent Delegation" below)

**What foreman CANNOT do:**
- ❌ NEVER edit files in other repos directly — delegate to a layer agent instead
- ❌ NEVER implement layer functionality from foreman's context
- ❌ NEVER fix violations directly — file fix-requests using `philosophy/workflows/fix-request.json`
- ❌ NEVER create scaffolding or automation until actual needs emerge 3+ times

**Scope discipline** (from `philosophy/scope.md`):
- **Need-to-know**: Only maintain what's immediately required for coordination
- **Minimal surface area**: Foreman's current structure (CLAUDE.md + test_observation_chain.py + README.md) is sufficient
- **Let usage guide evolution**: Don't add scripts/ or features until patterns demand them

**Validation status**: As of 2026-02-11, all paths verified correct, test_observation_chain.py works from new location.

## The Cascade Pattern (Essential)

**The cascade is Foreman's primary coordination mechanism** for resolving cross-layer issues:

### What is a Cascade?

When an issue in Layer N requires changes in Layer N-1 (or deeper):

```
User reports issue to Layer N
  ↓ (Layer N files fix-request to N-1)
Layer N BLOCKED, waiting for Layer N-1
  ↓ (Layer N-1 files fix-request to N-2)
Layer N-1 BLOCKED, waiting for Layer N-2
  ↓ (Layer N-2 can resolve)
Layer N-2 fixes, tests, closes (N-1 UNBLOCKS)
  ↓
Layer N-1 fixes, tests, closes (N UNBLOCKS)
  ↓
Layer N fixes, tests, closes (User notified)
  ↓
Foreman validates integration tests ✅
```

**Key properties:**
- Issues flow **down** (5→4→3)
- Each layer **blocks** until the one below resolves
- Fixes flow **up** (3✓→4✓→5✓)
- Each layer maintains **sovereignty** (no cross-layer edits)

### Foreman's Cascade Responsibilities

1. **Track cascades** — Create tracking file in `cascades/active/<cascade-id>.md`
2. **Monitor progress** — Update as layers file fix-requests and close issues
3. **Validate resolution** — Run integration tests when cascade completes
4. **Archive history** — Move to `cascades/completed/` or `cascades/failed/`
5. **Notify user** — Report completion or escalate failures

**Foreman does NOT:**
- ❌ File fix-requests on behalf of layers (sovereignty)
- ❌ Implement fixes (layers do that)
- ❌ Rush layers to close faster (respect blocking)

### Cascade Infrastructure

**Workflow definition**: `philosophy/workflows/cascade.json`
**Tracking system**: `foreman/cascades/` (active/, completed/, failed/)
**Template**: `foreman/cascades/template.md`
**Full guide**: `foreman/docs/cascade_pattern.md`
**Example**: `foreman/cascades/completed/2026-02-terrain-aware-locomotion.md`

### When to Create a Cascade

Create a cascade tracking file when:
- ✅ User reports issue to Layer N
- ✅ Layer N discovers it needs Layer N-1 changes
- ✅ Layer N files fix-request to Layer N-1

**Cascade ID format**: `YYYY-MM-<short-description>`

See `foreman/docs/cascade_pattern.md` for comprehensive guide.

---

## Foreman's Coordination Responsibilities

Foreman coordinates the **application of improvements** from `improvements/` to the **implementation layers** (`layer_3/`, `layer_4/`, `layer_5/`), ensuring:

1. **Architectural proposals are consistent with philosophy**
   - Read `improvements/IMPLEMENTATION_PLAN.md` for step-by-step execution plans
   - Read `improvements/ARCHITECTURE_JOURNEY.md` for design rationale
   - Verify proposals respect `philosophy/scope.md` and `philosophy/boundaries.md`

2. **Fix-requests respect layer sovereignty**
   - `improvements/fix-requests/` contains detailed specs for layer changes
   - Each fix-request targets ONE layer (e.g., `layer3_kinematic_state.md` → Layer 3)
   - Foreman NEVER files fix-requests on behalf of layers (layers are sovereign)
   - Foreman CAN validate that fix-requests follow `philosophy/workflows/fix-request.json` format

3. **Integration tests validate cross-layer contracts**
   - `test_observation_chain.py` validates FEP observation chain architecture
   - Tests check: N → N-1 layering, semantic naming, error propagation, latency
   - When improvements touch multiple layers, add/update integration tests
   - Integration tests are Foreman's primary artifact (beyond guidance)

4. **Version compatibility is understood**
   - Layer versions: layers_1_2 (v0.1.11), layer_3 (v2.1.0), layer_4 (v0.13.0), layer_5 (v0.6.0)
   - See `philosophy/workflows/repos.json` for repo URLs and status
   - When coordinating improvements, note which layer versions are targeted
   - Foreman doesn't enforce versions, but documents expectations

**What Foreman does NOT do:**
- ❌ Implement improvements directly in layer code
- ❌ File fix-requests on behalf of layers (each layer files their own)
- ❌ Make architectural decisions (that's in `improvements/`, informed by `philosophy/`)
- ❌ Control layer development (coordinate, don't control)

**Key insight**: Foreman is the **stage manager**, not the **director**. Improvements are the **script**, philosophy is the **playwright's intent**, and layers are the **actors**. Foreman sets context, ensures actors know their cues, and validates the performance meets the vision.

## What This Workspace Is

This is a **multi-layer robotics control stack** for Unitree SDK2 robots (B2, Go2, H1, G1). The workspace implements an 8-layer onion architecture where each layer translates from one control language to another, following the OSI network model pattern.

**Workspace structure**: Multi-repo workspace containing separate Git repositories for Layers 1-5, plus `philosophy/` (architecture docs), `training/` (offline ML), `improvements/` (planning), and `Assets/` (robot models). The `foreman/` repo coordinates across all repositories.

## Multi-Repo Workspace and Sovereignty

This workspace (`/Users/graham/code/robotics/`) contains **multiple sovereign Git repositories** that work together but remain separate:

- **foreman/** (THIS REPO): Workspace coordination — integration tests, cross-repo guidance. Each directory below is a **separate Git repository** with its own `.git/`, commit history, and VERSION.
- **Layers** (layers_1_2/, layer_3/, layer_4/, layer_5/): Each is a sovereign implementation with its own VERSION, CLAUDE.md, and API contract
- **Philosophy** (philosophy/): Source of truth for architecture, principles, and workflows
- **Training** (training/): Separate Git repository for offline ML training (HNN, RL)
- **Improvements** (improvements/): Architectural planning and research proposals
- **Assets** (Assets/): Shared MuJoCo robot models (layers_1_2/unitree_robots/ symlinks here)

**Key principle**: Each repository has **agency, scope, and responsibility**. Repositories may read other repos for reference but **NEVER edit files belonging to another repository**. To request changes, use `philosophy/workflows/fix-request.json`.

**Git structure**: The workspace root is NOT a git repo. Each subdirectory (foreman/, philosophy/, layers_1_2/, etc.) is its own independent Git repository. Use `cd <repo> && git status` to work with individual repos.

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

**Important**: All commands below assume you're working from the workspace root at `/Users/graham/code/robotics/`. If you're in the `foreman/` directory, use `cd ..` first.

```bash
# Activate Python environment (from workspace root)
source env/bin/activate

# Run the full stack (two terminals)
# Terminal 1: Start firmware (Layers 1-2)
cd layers_1_2 && python firmware_sim.py --robot b2

# Terminal 2: Run a controller (Layer 3)
cd layer_3 && python controller.py

# Or run tests for a specific layer
cd layer_3 && make test

# Run workspace-level integration tests
python foreman/test_observation_chain.py
```

## Workspace Structure

**Multi-repo workspace layout** (each directory is a separate Git repository):

```
/Users/graham/code/robotics/          # Workspace root (not a git repo)
├── foreman/             # THIS REPO: Workspace coordination
│   ├── CLAUDE.md        # This file (workspace-level guidance)
│   └── test_observation_chain.py  # Cross-layer integration test
│
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
└── env/                 # Python virtual environment
```

**Note**: `test_observation_chain.py` lives in `foreman/` and is run from workspace root as `python foreman/test_observation_chain.py`.

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

- **IMPLEMENTATION_PLAN.md**: Step-by-step plan for HNN/RL features (12-week timeline)
- **ARCHITECTURE_JOURNEY.md**: Historical record of design insights (how we discovered correct architecture)
- **LEARNING_ROADMAP.md**: Technical details for training (HNN, RL code examples)
- **fix-requests/**: Detailed implementation specs ready to file as GitHub issues
  - `layer3_kinematic_state.md` — KinematicState publisher for Layer 3
  - `layer4_dynamics_prediction.md` — Dynamics prediction for Layer 4
  - `layer5_fep_learning.md` — FEP-based learning for Layer 5

**Scope**: Documents proposals that span multiple layers. Individual layer changes go in `layer_X/plans/`.

**Foreman's role with improvements/**:
- ✅ Read improvements/ to understand architectural direction
- ✅ Validate that proposals respect philosophy/ principles (scope, boundaries, layering)
- ✅ Update integration tests when improvements touch multiple layers
- ✅ Track which improvements are planned vs implemented (via layer issue links in `improvements/README.md`)
- ❌ Never implement improvements directly in layers (sovereignty violation)
- ❌ Never make architectural decisions (that's improvements/' job, informed by philosophy/)

**Philosophy vs Improvements**:
- `philosophy/` = **established principles** (boundaries, scope, control, architecture model)
- `improvements/` = **architectural proposals** (what to build next, how it fits principles)
- Foreman ensures improvements **respect** philosophy, not the other way around

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

## Implementation Status Tracking

Foreman tracks cross-layer implementation status from `improvements/README.md`:

### Current State (as of 2026-02-11)

**FEP terrain-aware locomotion: ✅ COMPLETE**

| Layer | Issue | Feature | Status |
|-------|-------|---------|--------|
| Layer 3 | [#11](https://github.com/Pangeon-Robotics/layer_3/issues/11) | KinematicState publisher | ✅ v2.1.0 |
| Layer 4 | [#19](https://github.com/Pangeon-Robotics/layer_4/issues/19) | Simple terrain estimation | ✅ v0.13.0 |
| Layer 5 | [#2](https://github.com/Pangeon-Robotics/layer_5/issues/2) | Terrain-aware gait selection | ✅ v0.6.0 |

**Key insight**: Rule-based implementation complete. Used sensor heuristics instead of HNN (Layer 4 #18 rejected - learning violated stateless boundary).

### Next Phase: Adding Learning

**Planned improvements from `improvements/IMPLEMENTATION_PLAN.md`:**

1. **Phase 1: Training Infrastructure** (Week 1-2)
   - Location: `training/` repo (separate from layers)
   - HNN dynamics model (JAX/Flax)
   - RL locomotion policy (PyTorch/SB3)
   - Status: Planned, not implemented

2. **Phase 2: HNN Training and Layer 4 Deployment** (Week 3-5)
   - HNN learns terrain dynamics offline
   - Layer 4 receives frozen HNN model for prediction
   - Respects stateless constraint (inference only, no online learning)

3. **Phase 3: RL Policy Training and Layer 5 Deployment** (Week 6-10)
   - RL learns gait selection offline
   - Layer 5 receives frozen policy for gait selection
   - Respects sequence nature (Layer 5 is stateful, can use RL)

**Foreman's tracking responsibilities**:
- ✅ Maintain this status section in CLAUDE.md as improvements are implemented
- ✅ Verify integration tests pass after each phase
- ✅ Document version compatibility requirements
- ❌ Do NOT implement phases (layers implement, training/ trains)
- ❌ Do NOT decide what to build next (improvements/ decides)

See `improvements/README.md` and `improvements/IMPLEMENTATION_PLAN.md` for detailed plans.

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

| Workflow | File | Foreman's Role |
|----------|------|----------------|
| **P cascade** | `cascade.json` | **Track and validate** — Create cascade tracking file, monitor progress, run integration tests when complete. Do NOT file fix-requests or implement fixes. |
| **P fix-request** | `fix-request.json` | **Guide only** — Foreman NEVER files fix-requests on behalf of layers. Each layer files their own. Foreman can validate format. |
| **P resolve-issues** | `resolve-issues.json` | **Self-apply** — Foreman runs this on its own issues in the foreman repo. |
| **P compliance** | `compliance.json` | **Self-apply** — Foreman checks itself against philosophy docs. Can also guide layers through this workflow. |

**Key principle**: Foreman coordinates workflows but **never executes them on behalf of other repos**. Each repo is sovereign.

**The Cascade Pattern** (see full guide in `docs/cascade_pattern.md`):
1. **User reports issue** to Layer N
2. **Layer N discovers** it needs Layer N-1 changes
3. **Cascade flows down**:
   - Layer N files fix-request to Layer N-1 (Layer N BLOCKED)
   - Layer N-1 files fix-request to Layer N-2 (Layer N-1 BLOCKED)
   - Continue until resolving layer reached
4. **Cascade flows up**:
   - Layer N-2 fixes, closes (Layer N-1 UNBLOCKS)
   - Layer N-1 fixes, closes (Layer N UNBLOCKS)
   - Layer N fixes, closes (User notified)
5. **Foreman validates** integration tests ✅

**Foreman's cascade tracking**:
- Create: `cascades/active/<cascade-id>.md` (using template)
- Monitor: Update as layers file fix-requests and close issues
- Validate: Run integration tests when cascade completes
- Archive: Move to `cascades/completed/` or `cascades/failed/`

**Example**: See `cascades/completed/2026-02-terrain-aware-locomotion.md` for a complete cascade (Layer 5→4→3, then 3✓→4✓→5✓).

## Working Across Layers

When Claude operates in this workspace:

### If in foreman/ (THIS REPO):
1. **Read philosophy/scope.md and philosophy/boundaries.md FIRST** — These are your north star for operating in the workspace
2. **Coordinate, don't control** — Provide guidance and run integration tests; never implement layer features
3. **Respect all repo sovereignty** — Read other repos for understanding, but NEVER edit their files
4. **Minimal additions** — Before adding files/features to foreman, ask: "Does this coordinate, navigate, or test integration?"

### Agent Delegation for Layer Work (Essential)

**When the user asks Foreman to do work in a layer, Foreman MUST spawn a subagent that operates within that layer.** Foreman never reaches into a layer to edit files directly — it delegates to an agent that becomes a native worker in that layer's context.

**Every layer agent prompt MUST include:**
1. `cd` to the layer's directory first (e.g., `cd /home/graham/code/robotics/layer_5`)
2. Read the layer's `CLAUDE.md` before doing anything
3. The specific task to accomplish
4. Sovereignty reminder: never edit files outside this layer; if you need changes in a lower layer, report back

**Workflow routing for layer agents:**
- **Resolve issues** → agent follows `../philosophy/workflows/resolve-issues.json`
- **Run compliance** → agent follows `../philosophy/workflows/compliance.json`
- **General work** → agent reads layer's CLAUDE.md and proceeds

**When Foreman needs a layer to fix something** (e.g., cascade dependency):
- Foreman follows `philosophy/workflows/fix-request.json` to file the request
- Foreman does NOT spawn an agent to make the fix — it files a request, respecting sovereignty

**Rules:**
- ✅ One agent per layer — never spawn one agent across multiple layers
- ✅ Agent reads the **layer's** CLAUDE.md, not foreman's
- ❌ Never have Foreman edit layer files directly, even "just a small fix"
- ❌ Never give the agent instructions that contradict the layer's CLAUDE.md

### If in a specific layer directory (layer_3/, layer_4/, etc.):
1. **Read that layer's CLAUDE.md first** — it defines scope, boundaries, and vocabulary
2. **Respect sovereignty** — never edit files in other layers
3. **Use fix-request workflow** — if Layer N needs changes in Layer N-1, file `P fix-request`
4. **Stay in vocabulary** — describe work using only that layer's concepts

### If in root directory or cross-layer work:
1. **Read philosophy/scope.md** — understand need-to-know and vocabulary compression
2. **Read foreman/CLAUDE.md** — this file provides workspace navigation
3. **Identify which layer owns the work** — don't mix layer concerns
4. **Route to appropriate layer** — each layer has its own agent context

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

**foreman/test_observation_chain.py** — Workspace-level test validating the FEP Observation Chain:

```bash
# From workspace root
python foreman/test_observation_chain.py
```

Tests:
- Strict N → N-1 layering (no layer skipping)
- Semantic naming (KinematicState, DynamicState, BehavioralState)
- FEP "explain away" pattern (error propagation upward)
- Latency constraints across layers

This is a **cross-layer integration test** maintained in the foreman repo. Individual layers have their own unit tests in their own directories.

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

## Foreman Evolution

**Current structure** (sufficient as of 2026-02-11):
```
foreman/
├── CLAUDE.md                    # Workspace guidance (this file)
├── README.md                    # Foreman scope documentation
├── test_observation_chain.py    # Cross-layer FEP integration test
└── .git/                        # Git repository
```

**When to add features** (from `philosophy/scope.md`):
- **Compression test**: Can you remove it and foreman still does its job? If yes, don't add it.
- **3+ usage rule**: If you find yourself doing the same cross-repo task 3+ times, consider automation
- **Capability-oriented**: Ask "what capability does this enable?" not "what tool should I add?"

**Potential future additions** (only when actual needs emerge):
```
scripts/                         # Add only when patterns demand it
├── workspace_health.sh          # If you check git status across repos 3+ times
├── version_report.sh            # If you query VERSION files 3+ times
└── dependency_check.sh          # If you verify deps 3+ times
```

**Golden rule**: Foreman is brand new (created 2026-02-11). Let usage patterns guide evolution. Don't create scaffolding for hypothetical needs.

## Additional Resources

- **[philosophy/gotchas.md](philosophy/gotchas.md)** — Lessons from 40+ postmortems
- **[philosophy/training.md](philosophy/training.md)** — Dense reward shaping, RL methodology
- **[philosophy/deployment.md](philosophy/deployment.md)** — Two-tier simulation strategy
- Layer-specific `ARCHITECTURE.md` files for implementation details
- Layer-specific `docs/api.md` files for public interfaces

---

## Changelog

### 2026-02-11 (Latest)
**Added — Cascade Pattern Codification**:
- **"The Cascade Pattern (Essential)"** section — primary coordination mechanism
- `philosophy/workflows/cascade.json` — formal workflow definition
- `cascades/` tracking system — active/, completed/, failed/ directories
- `cascades/template.md` — template for tracking new cascades
- `cascades/completed/2026-02-terrain-aware-locomotion.md` — real example
- `docs/cascade_pattern.md` — comprehensive 500+ line guide
- Enhanced "Workflows" section — cascade tracking responsibilities

**Previously added**:
- "Foreman's Coordination Responsibilities" section — explicit role in applying improvements
- "Implementation Status Tracking" section — tracks progress from `improvements/README.md`
- Enhanced "Improvements (Architectural Planning)" section — role with fix-requests
- Explicit philosophy/ vs improvements/ distinction (principles vs proposals)

**Key insights**:
- Foreman is stage manager (sets context, validates), not director (doesn't implement or decide)
- Cascades are the primary mechanism for cross-layer issue resolution
- Issues flow down (5→4→3), fixes flow up (3✓→4✓→5✓), Foreman validates

### 2026-02-11 (Initial)
- Foreman repo created
- Initial CLAUDE.md with workspace structure, layer architecture, and critical principles
- `test_observation_chain.py` integration test established

---

**Foreman repo created**: 2026-02-11 (brand new — expect refinement based on usage)
**Last updated**: 2026-02-11 (cascade pattern codified)
**Architecture version**: 8-layer onion model
**Workspace**: Multi-repo at `/Users/graham/code/robotics/`
**This file location**: `foreman/CLAUDE.md` (provides workspace-level guidance)
**Philosophy docs**: Read `philosophy/scope.md` and `philosophy/boundaries.md` as north star
**Applies to**: Unitree SDK2 robots (B2, Go2, G1, H1)
