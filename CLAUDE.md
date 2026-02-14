# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Foreman** is workspace coordination for a multi-layer robotics control stack (Unitree SDK2 robots: B2, Go2, H1, G1). It provides cross-repo guidance, integration tests, and cascade tracking. It does NOT implement any layer functionality.

**Workspace root**: The parent directory of `foreman/` (multi-repo, not itself a git repo). All sibling directories (`layers_1_2/`, `layer_3/`, etc.) are at the same level.

**Foreman is a stage manager, not a director.** It coordinates, navigates, and validates integration across sovereign layer repositories. It never implements, controls, or makes architectural decisions for layers.

## Foreman's Boundaries

**CAN do**: Read any repo for understanding; maintain integration tests; update guidance in this repo; coordinate workflows; delegate layer work to subagents

**CANNOT do**: Edit files in other repos (delegate instead); implement layer functionality; file fix-requests on behalf of layers; create scaffolding until patterns emerge 3+ times

**Vocabulary**: "workspace", "coordination", "navigation", "integration", "sovereignty" — never layer-specific terms like "kinematics", "gaits", "IK"

## The 8-Layer Architecture

| Layer | Directory | Status | Purpose |
|-------|-----------|--------|---------|
| 1 | `layers_1_2/` | Complete | Motor Drivers / Physics (MuJoCo) |
| 2 | `layers_1_2/` | Complete | Firmware PD Control |
| 3 | `layer_3/` | Complete | Inverse Kinematics |
| 4 | `layer_4/` | Complete | Cartesian Positions (gait params to foot positions) |
| 5 | `layer_5/` | In Progress | Locomotion (motion commands to gait parameters) |
| 6-8 | Not implemented | | Waypoints, Mission Planning, Application UI |

Layers 1-4 are **instant** (stateless per-timestep). Layers 5-8 are **sequences** (stateful). The DDS boundary sits between Layers 2-3. Commands flow down (5 to 1), observations flow up (1 to 5).

## Multi-Repo Workspace

Each subdirectory is a **separate Git repo** with its own `.git/`, CLAUDE.md, and VERSION:

- **layers_1_2/**, **layer_3/**, **layer_4/**, **layer_5/** — Sovereign layer implementations
- **philosophy/** — Source of truth for architecture, principles, workflows
- **training/** — Offline ML training (HNN, RL) — separate from the layer stack
- **improvements/** — Architectural planning and research proposals
- **Assets/** — Shared MuJoCo robot models (`layers_1_2/unitree_robots/` symlinks here)

Each repo has **agency, scope, and responsibility**. Never edit files belonging to another repository. To request changes, use `philosophy/workflows/fix-request.json`.

## Commands

### Environment Setup
```bash
# From workspace root (parent of foreman/)
source env/bin/activate

# Install per-layer deps
cd layers_1_2 && make install
cd ../layer_3 && make install
cd ../layer_4 && make install
cd ../layer_5 && make install

# Unitree SDK2 (vendored, not on PyPI)
pip install -e unitree_sdk2_python/
```

### Running the Full Stack (Two Terminals)
```bash
# Terminal 1: Firmware (Layers 1-2)
cd layers_1_2 && python firmware_sim.py --robot b2

# Terminal 2: Controller (Layer 3+)
cd layer_3 && python controller.py
```

### Per-Layer Commands
Each layer uses `make install`, `make test`, `make run`. Example:
```bash
cd layer_3 && make test                    # Run all layer 3 tests
cd layer_3 && pytest tests/test_controller.py -v  # Single test file
```

### Foreman's Own Tests
```bash
# Integration test: FEP observation chain (from workspace root)
python foreman/test_observation_chain.py
python foreman/test_observation_chain.py --demo  # With simulation demo
```

### Target Game Demo
Cross-layer integration demo that spawns random targets and drives the robot to them:
```bash
# From workspace root
python -m foreman.demos.target_game --robot b2 --targets 5
python -m foreman.demos.target_game --robot b2 --targets 3 --headless
python -m foreman.demos.target_game --robot b2 --seed 42  # Reproducible
```
Runs Layers 1-5 together. Uses `scene_target.xml` with a mocap target marker. The `__main__.py` patches cross-layer config namespace collisions at import time.

### Troubleshooting
```bash
# Kill zombie firmware processes (common after interrupted runs)
pkill -9 -f "firmware_sim.py"
```

## The Cascade Pattern

Foreman's primary coordination mechanism for cross-layer issue resolution. When Layer N needs changes in Layer N-1:

1. Issues flow **down** (5 to 4 to 3) via fix-requests, each layer blocks
2. Fixes flow **up** (3 done, 4 unblocks, 5 unblocks), sovereignty maintained
3. Foreman **tracks** (creates `cascades/active/<id>.md`), **validates** (runs integration tests), and **archives** (`cascades/completed/` or `cascades/failed/`)

Foreman does NOT file fix-requests or implement fixes on behalf of layers.

**Infrastructure**: `cascades/template.md` (template), `docs/cascade_pattern.md` (full guide), `cascades/completed/2026-02-terrain-aware-locomotion.md` (real example), `philosophy/workflows/cascade.json` (workflow definition)

## Agent Delegation for Layer Work

**When the user asks to do work in a layer, spawn a subagent within that layer.** Never edit layer files directly from foreman context.

Every layer agent prompt MUST:
1. `cd` to the layer's directory (workspace root + layer dir, e.g., `cd ../layer_5` from foreman)
2. Read the layer's CLAUDE.md before anything else
3. Include the specific task
4. Remind: never edit files outside this layer; report back if lower-layer changes needed

Rules: one agent per layer; agent follows the layer's CLAUDE.md, not foreman's.

## Workflows

When the user says "P <workflow>", execute from `philosophy/workflows/`:

| Command | File | Foreman's Role |
|---------|------|----------------|
| P cascade | `cascade.json` | Track and validate (don't file fix-requests) |
| P fix-request | `fix-request.json` | Guide only (layers file their own) |
| P resolve-issues | `resolve-issues.json` | Self-apply to foreman repo |
| P compliance | `compliance.json` | Self-apply or guide layers |

## Critical Principles

### Joint vs Actuator Order (CRITICAL)
MuJoCo and DDS use different orderings. **Forgetting this causes immediate robot collapse.**
- MuJoCo body tree: FL, FR, RL, RR
- DDS actuators: FR, FL, RR, RL
- Permutation: `[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]` (self-inverse)

### DDS Communication
- **Never call `ChannelFactoryInitialize()`** — buffer overflow in CycloneDDS 0.10.4
- Use `dds_init(domain_id=1, interface="lo")` from `dds.py`
- Domain IDs: 1 = simulation, 0 = real robot
- CRC required: call `stamp_cmd(cmd)` before every `pub.Write(cmd)`

### Hardware Abstraction
**Allowed** (available on real robot): joint encoders (`motor_state[i].q`, `.dq`, `.tau_est`), IMU, foot contact forces

**Forbidden** (simulation-only): `data.qpos[0:3]` (GPS), `data.xpos` (mocap), contact geometry, `import mujoco` in Layer 3+

### Layered Abstraction
- Never use `if simulation:` branches — controller code must be identical for real and simulated robots
- Each layer uses only sensors/data available on real hardware

### Simulation Gains
Test gains: kp=500, kd=25. Production gains (kp=2500) cause oscillation in simulation.

## Dependencies

- Python 3.12 (Ubuntu 24.04)
- MuJoCo 3.3.0+
- CycloneDDS 0.10.4 (**must be this version**, 0.10.5+ has buffer overflow)
- NumPy 2.4.1+
- Robot models: `Assets/unitree_robots/` (download from HuggingFace unitree-robotics/unitree_robots)

## Key References

- `philosophy/scope.md` and `philosophy/boundaries.md` — North star for workspace operation
- `philosophy/architecture.md` — Canonical 8-layer model
- `philosophy/gotchas.md` — Lessons from 40+ postmortems
- `improvements/IMPLEMENTATION_PLAN.md` — Next phase planning (HNN/RL)
- Each layer's own `CLAUDE.md` — Authoritative for that layer's development
