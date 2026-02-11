# Foreman

Workspace coordination for the Unitree robotics control stack.

## What This Repo Is

**Foreman** coordinates work across multiple sovereign layer repositories. It provides:

- **Workspace guidance** (CLAUDE.md) — How to navigate and work across all layers
- **Integration tests** — Cross-layer validation (e.g., `test_observation_chain.py`)
- **Coordination scripts** — Tools for managing the multi-repo workspace

**Not a layer**: Foreman doesn't implement any layer of the 8-layer architecture. It's the orchestration layer that helps developers and agents work effectively across the stack.

## Quick Start

```bash
# From the parent workspace directory:
cd /Users/graham/code/robotics

# Read workspace guidance
cat foreman/CLAUDE.md

# Run cross-layer integration tests
python foreman/test_observation_chain.py
```

## Contents

- **CLAUDE.md** — Root-level guidance for Claude Code when working across all repositories
- **test_observation_chain.py** — Integration test validating FEP Observation Chain architecture
- **README.md** — This file

## Repository Structure Context

The foreman repo sits at the coordination level:

```
/Users/graham/code/robotics/
├── foreman/              # This repo (workspace coordination)
│   ├── CLAUDE.md         # Cross-repo guidance
│   └── test_observation_chain.py
│
├── philosophy/           # Architecture docs (source of truth)
├── layers_1_2/           # Layers 1-2 implementation
├── layer_3/              # Layer 3 implementation
├── layer_4/              # Layer 4 implementation
├── layer_5/              # Layer 5 implementation
├── training/             # Offline ML training
├── improvements/         # Architectural planning
└── Assets/               # Robot models
```

Each layer is a sovereign Git repository. Foreman coordinates but doesn't control.

## Scope

**Foreman does:**
- Provide workspace navigation guidance
- Run cross-layer integration tests
- Coordinate multi-repo workflows

**Foreman does NOT:**
- Implement any layer functionality
- Edit files in other repositories (sovereignty rule)
- Contain layer-specific logic

See `philosophy/scope.md` for information economy principles.

## Relationship to Other Repos

| Repo | Scope | Relationship to Foreman |
|------|-------|------------------------|
| **philosophy/** | Architecture docs, principles | Foreman references philosophy as source of truth |
| **layers_1_2/** to **layer_5/** | Layer implementations | Foreman coordinates but never edits |
| **training/** | Offline ML training | Independent; foreman doesn't coordinate training |
| **improvements/** | Architectural planning | Strategic planning; foreman is operational |
| **Assets/** | Robot models | Shared resource; foreman doesn't manage |

## Development

```bash
# Add new coordination scripts
cd foreman
git add <script>
git commit -m "Add coordination script"
git push origin main

# Update workspace guidance
vim CLAUDE.md
git commit -m "Update workspace guidance"
git push origin main
```

## Philosophy Alignment

From `philosophy/scope.md`:
- **Need-to-know**: Foreman provides only what's needed for cross-repo coordination
- **Minimal vocabulary**: Foreman speaks in terms of "layers", "workflows", "navigation"
- **Capabilities not tools**: Foreman enables the capability of "coordinating sovereign repositories"

Foreman is a **capability** (workspace coordination), not a collection of tools.

---

**Created**: 2026-02-11
**Status**: Active
**Repo**: https://github.com/Pangeon-Robotics/foreman
