# Foreman

Workspace coordination for the Unitree robotics control stack.

## What This Repo Is

**Foreman** coordinates work across multiple sovereign layer repositories. It provides:

- **Workspace guidance** (CLAUDE.md) — How to navigate and work across all layers
- **Cascade tracking** — Monitor cross-layer issue resolution flows
- **Integration tests** — Cross-layer validation (e.g., `test_observation_chain.py`)
- **Coordination workflows** — Tools for managing the multi-repo workspace

**Not a layer**: Foreman doesn't implement any layer of the 8-layer architecture. It's the orchestration layer that helps developers and agents work effectively across the stack.

## Quick Start

```bash
# From the parent workspace directory (one level above foreman/):
cd ..

# Read workspace guidance
cat foreman/CLAUDE.md

# Understand the cascade pattern (essential!)
cat foreman/docs/cascade_pattern.md

# Run cross-layer integration tests
python foreman/test_observation_chain.py

# Track a new cascade
cp foreman/cascades/template.md foreman/cascades/active/2026-XX-description.md
# Fill in cascade details as layers file fix-requests
```

## Contents

### Core Files
- **CLAUDE.md** — Root-level guidance for Claude Code when working across all repositories
- **README.md** — This file
- **test_observation_chain.py** — Integration test validating FEP Observation Chain architecture

### Cascade Pattern (Essential)
- **cascades/** — Tracking system for cross-layer issue resolution
  - `README.md` — Cascade tracking documentation
  - `template.md` — Template for new cascades
  - `active/` — Currently in-progress cascades
  - `completed/` — Successfully resolved cascades (includes real example)
  - `failed/` — Failed cascades requiring escalation
- **docs/cascade_pattern.md** — Comprehensive guide to the cascade pattern

## Repository Structure Context

The foreman repo sits at the coordination level:

```
<workspace-root>/
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
- Track cascade patterns (cross-layer issue resolution)
- Run cross-layer integration tests
- Coordinate multi-repo workflows
- Validate integration when cascades complete

**Foreman does NOT:**
- Implement any layer functionality
- File fix-requests on behalf of layers (sovereignty)
- Edit files in other repositories (sovereignty rule)
- Contain layer-specific logic
- Rush layers to close issues faster

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
