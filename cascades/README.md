# Cascade Tracking

This directory tracks active and historical **cascades** — cross-layer issue resolution flows.

## What is a Cascade?

A cascade occurs when an issue in Layer N requires changes in Layer N-1 (or deeper) to resolve:

```
User reports issue to Layer 5
  ↓ (Layer 5 attempts, discovers needs Layer 4)
Layer 5 files fix-request to Layer 4 (Layer 5 BLOCKED)
  ↓ (Layer 4 attempts, discovers needs Layer 3)
Layer 4 files fix-request to Layer 3 (Layer 4 BLOCKED)
  ↓ (Layer 3 can resolve)
Layer 3 fixes, tests, closes (Layer 4 UNBLOCKS)
  ↓
Layer 4 fixes, tests, closes (Layer 5 UNBLOCKS)
  ↓
Layer 5 fixes, tests, closes (User notified)
  ↓
Foreman validates integration tests ✅
```

**Key properties:**
- Issues flow **down** (5→4→3)
- Each layer **blocks** waiting for the one below
- Fixes flow **up** (3✓→4✓→5✓)
- Each layer maintains **sovereignty** (no cross-layer edits)

## Directory Structure

```
cascades/
├── README.md              # This file
├── template.md            # Template for tracking a new cascade
├── active/                # Currently in-progress cascades
│   └── <cascade-id>.md
├── completed/             # Successfully resolved cascades
│   └── <cascade-id>.md
└── failed/                # Cascades that failed to resolve
    └── <cascade-id>.md
```

## Creating a New Cascade

When a cascade is initiated:

1. Copy `template.md` to `active/<cascade-id>.md`
2. Fill in cascade details (initiating layer, issue links)
3. Update status as layers file fix-requests
4. Track blocking relationships
5. Update as layers resolve
6. Move to `completed/` or `failed/` when done

**Cascade ID format**: `YYYY-MM-<short-description>`
Example: `2026-02-terrain-aware-locomotion`

## Cascade States

| State | Meaning |
|-------|---------|
| **initiated** | User reported issue to Layer N |
| **cascading** | Fix-requests flowing down through layers |
| **resolving** | Bottom layer fixing, others still blocked |
| **complete** | All layers resolved, integration tests pass |
| **failed** | Cascade failed to resolve (escalate to user) |

## Foreman's Responsibilities

As the cascade tracker, Foreman:

1. **Creates tracking file** when cascade is initiated
2. **Updates status** as layers file fix-requests and close issues
3. **Monitors blocking** relationships (who's waiting on whom)
4. **Validates resolution** by running integration tests
5. **Notifies user** when cascade completes or fails
6. **Archives** to completed/ or failed/ directories

**Foreman does NOT:**
- ❌ Implement fixes (layers do that)
- ❌ File fix-requests on behalf of layers (sovereignty)
- ❌ Decide what to build (improvements/ decides)
- ❌ Skip layers to speed up cascade (breaks sovereignty)

## Example Cascade: Terrain-Aware Locomotion

See `completed/2026-02-terrain-aware-locomotion.md` for a real example of the cascade pattern in action.

## Integration with Workflows

The cascade pattern composes existing workflows:

- **fix-request.json** — Each layer files fix-requests using this
- **resolve-issues.json** — Each layer resolves using this
- **cascade.json** — Orchestrates the full pattern across layers

See `../philosophy/workflows/cascade.json` for the formal definition.

---

**Last updated**: 2026-02-11
**Status**: Active tracking system
