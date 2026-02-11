# Foreman Repository Structure

**Created**: 2026-02-11
**Purpose**: Workspace coordination for multi-layer robotics control stack

---

## Directory Tree

```
foreman/
├── CLAUDE.md                    # Workspace guidance (essential reading)
├── README.md                    # Repository overview
├── STRUCTURE.md                 # This file (structure documentation)
│
├── cascades/                    # CASCADE TRACKING (essential)
│   ├── README.md                # Tracking system documentation
│   ├── template.md              # Template for new cascades
│   ├── active/                  # In-progress cascades
│   ├── completed/               # Successfully resolved cascades
│   │   └── 2026-02-terrain-aware-locomotion.md  # Real example
│   └── failed/                  # Failed cascades (escalate to user)
│
├── docs/                        # Documentation
│   └── cascade_pattern.md       # Comprehensive cascade guide (500+ lines)
│
├── test_observation_chain.py    # Integration test (FEP observation chain)
│
└── .git/                        # Git repository
```

---

## File Purposes

### Core Coordination Files

**CLAUDE.md** (workspace-level guidance)
- Foreman's identity and boundaries
- The Cascade Pattern (essential section)
- Multi-repo workspace structure
- 8-layer architecture overview
- Critical principles and workflows
- **Read this first** when working as Foreman

**README.md** (public-facing overview)
- What Foreman is (and isn't)
- Quick start guide
- Relationship to other repos
- Development workflows

**STRUCTURE.md** (this file)
- Repository structure documentation
- File purposes and relationships
- When things should be added

---

## Cascade Infrastructure (Essential)

### Why Cascades Are Central

The **cascade pattern** is Foreman's primary coordination mechanism. When an issue in Layer N requires changes in Layer N-1 (or deeper), the cascade pattern ensures:

- Issues flow **down** through layers (5→4→3)
- Each layer **blocks** until the one below resolves
- Fixes flow **up** sequentially (3✓→4✓→5✓)
- Sovereignty is maintained (no cross-layer edits)
- Integration is validated (Foreman runs tests)

### Cascade Directory Structure

**cascades/README.md**
- Overview of cascade pattern
- Directory structure explanation
- Foreman's tracking responsibilities
- When to create cascades

**cascades/template.md**
- Template for tracking new cascades
- Sections: Overview, Flow, Timeline, Tests, Notes
- Copy to `active/<cascade-id>.md` when cascade initiates

**cascades/active/**
- Currently in-progress cascades
- One file per cascade: `YYYY-MM-short-description.md`
- Updated as layers file fix-requests and close issues
- Moved to `completed/` or `failed/` when done

**cascades/completed/**
- Successfully resolved cascades (historical record)
- Example: `2026-02-terrain-aware-locomotion.md`
- Shows full cascade flow: down then up
- Includes version bumps and integration test results

**cascades/failed/**
- Cascades that could not resolve
- Requires user escalation
- Documents what went wrong and why

---

## Documentation

### docs/cascade_pattern.md (Comprehensive Guide)

A 500+ line guide covering:
- What is a cascade? (concept and diagram)
- Why cascades matter (sovereignty, boundaries, traceability)
- Anatomy of a cascade (roles, states, phases)
- Running a cascade (step-by-step for each role)
- Real example (terrain-aware locomotion)
- Anti-patterns (what NOT to do)
- Checklists and success metrics

**When to read**:
- Before creating your first cascade
- When explaining cascades to team members
- When debugging a stuck cascade
- As reference during cascade execution

---

## Integration Tests

### test_observation_chain.py

Validates FEP Observation Chain architecture:
- Strict N → N-1 layering (no skipping)
- Semantic naming (KinematicState, DynamicState, BehavioralState)
- FEP "explain away" pattern
- Latency constraints across layers

**When to run**:
- After a cascade completes (validation step)
- When making changes to Foreman
- As part of CI/CD (future)

---

## Philosophy Integration

### philosophy/workflows/cascade.json (Owned by Philosophy Repo)

Formal workflow definition for the cascade pattern. Located in the philosophy repo because:
- Cascades are a **cross-repo coordination workflow**
- Philosophy repo owns workflow definitions
- All layers reference the same workflow definition

**Note**: This is the ONLY file Foreman created outside its own repo (sovereignty exception for coordination workflows).

---

## When to Add New Files

### Add to foreman/ when:
✅ It coordinates across multiple repos
✅ It tracks cross-layer patterns
✅ It provides workspace-level guidance
✅ It validates integration contracts

### Do NOT add to foreman/ when:
❌ It implements layer functionality
❌ It's layer-specific (belongs in that layer's repo)
❌ It's architectural principles (belongs in philosophy/)
❌ It's planning/proposals (belongs in improvements/)

### Golden Rule

From `philosophy/scope.md`:
- **Compression test**: Can you remove it and Foreman still does its job? If yes, don't add it.
- **3+ usage rule**: If you do the same cross-repo task 3+ times, consider automation.
- **Capability-oriented**: Ask "what capability does this enable?" not "what tool should I add?"

---

## Foreman's Evolution

**Current structure is sufficient** (as of 2026-02-11):
- Core files: CLAUDE.md, README.md, test_observation_chain.py ✅
- Cascade tracking: Full infrastructure established ✅
- Documentation: cascade_pattern.md comprehensive guide ✅

**Potential future additions** (only when patterns demand):
- `scripts/workspace_health.sh` — If checking git status across repos 3+ times
- `scripts/version_report.sh` — If querying VERSION files 3+ times
- Additional integration tests — When new cross-layer contracts emerge
- Cascade analytics — If tracking patterns across many cascades

**Let usage guide growth** — Don't add scaffolding for hypothetical needs.

---

## Relationship to Other Repos

| Repo | Foreman's Role |
|------|----------------|
| **philosophy/** | References as source of truth, created cascade.json workflow |
| **improvements/** | Tracks implementation status, understands proposals |
| **layers_1_2/** to **layer_5/** | Coordinates but never edits |
| **training/** | No coordination needed (independent) |
| **Assets/** | No coordination needed (shared resource) |

---

## Version History

### v1.0.0 (2026-02-11)
- Initial repository structure
- CLAUDE.md workspace guidance
- test_observation_chain.py integration test
- Cascade pattern codification
- Full tracking infrastructure
- Comprehensive documentation

---

**Status**: Active
**Repo**: https://github.com/Pangeon-Robotics/foreman
**Philosophy**: Read `philosophy/scope.md` and `philosophy/boundaries.md` as north star
