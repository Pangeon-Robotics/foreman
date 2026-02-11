# The Cascade Pattern

A comprehensive guide to cross-layer issue resolution in the Unitree robotics control stack.

---

## What is a Cascade?

A **cascade** is a coordinated pattern for resolving issues that span multiple layers:

1. **Issues flow down** through layers (Layer N → N-1 → N-2...)
2. **Each layer blocks** waiting for the layer below to resolve
3. **Fixes flow up** as layers complete (Layer 3✓ → 4✓ → 5✓)
4. **Sovereignty is maintained** — no layer edits another layer's code

```
┌─────────────────────────────────────────────────────┐
│                  User reports issue                  │
│                   to Layer 5                         │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Layer 5 attempts resolution  │
        │  Discovers needs Layer 4      │
        └───────────────┬───────────────┘
                        │ fix-request
                        ▼
        ┌───────────────────────────────┐
        │  Layer 4 attempts resolution  │
        │  Discovers needs Layer 3      │
        │  [Layer 5 BLOCKED]            │
        └───────────────┬───────────────┘
                        │ fix-request
                        ▼
        ┌───────────────────────────────┐
        │  Layer 3 can resolve          │
        │  Everything in scope          │
        │  [Layer 4, 5 BLOCKED]         │
        └───────────────┬───────────────┘
                        │ fixes, tests, closes
                        ▼
        ┌───────────────────────────────┐
        │  Layer 4 unblocks             │
        │  Completes work, closes       │
        │  [Layer 5 BLOCKED]            │
        └───────────────┬───────────────┘
                        │ fixes, tests, closes
                        ▼
        ┌───────────────────────────────┐
        │  Layer 5 unblocks             │
        │  Completes work, closes       │
        └───────────────┬───────────────┘
                        │ notifies user
                        ▼
        ┌───────────────────────────────┐
        │  Foreman validates            │
        │  Integration tests pass ✅     │
        └───────────────────────────────┘
```

---

## Why Cascades Matter

### Preserves Sovereignty

Each layer maintains **ownership** of its code. Layer N cannot directly edit Layer N-1's files — it must file a fix-request and wait.

### Enforces Boundaries

The cascade pattern forces each layer to **articulate what it needs** from the layer below. This surfaces:
- Scope violations (asking for something outside N-1's boundaries)
- Temporal violations (instant layer requesting stateful behavior)
- Feature creep (asking for too much)

### Maintains Contract Discipline

Because layers can only communicate through **defined interfaces** (fix-requests), the cascade ensures:
- Changes are reviewed by the owning layer
- API contracts are respected
- Dependencies flow in the correct direction (downward)

### Provides Traceability

Foreman tracks cascades, creating a **historical record** of:
- What was requested and why
- How the issue flowed through layers
- Which versions resolved what
- Integration test validation

---

## Anatomy of a Cascade

### Roles

| Role | Responsibility |
|------|----------------|
| **Initiating Layer** | Discovers the issue, files first fix-request |
| **Intermediate Layers** | Delegates further down if needed, blocks until resolved |
| **Resolving Layer** | Actually implements the fix (deepest layer) |
| **Foreman** | Tracks cascade, validates integration, notifies user |

### States

| State | Meaning |
|-------|---------|
| `initiated` | User reported issue, Layer N analyzing |
| `cascading` | Fix-requests flowing down through layers |
| `resolving` | Bottom layer implementing fix, others blocked |
| `complete` | All layers resolved, tests pass ✅ |
| `failed` | Cascade could not resolve (escalate to user) |

### Phases

#### 1. Initiation

User (as foreman) reports an issue to Layer N:

```
User: "Layer 5, implement terrain-aware gait selection"
Layer 5: "I need terrain estimates from Layer 4"
```

#### 2. Downward Cascade

Each layer analyzes and delegates:

```
Layer 5 → files fix-request to Layer 4 (BLOCKED)
Layer 4 → files fix-request to Layer 3 (BLOCKED)
Layer 3 → "I can fix this, it's in my scope"
```

#### 3. Resolution

The deepest layer fixes and closes:

```
Layer 3:
  - Implements KinematicState publisher
  - All tests pass
  - Commits and pushes
  - Bumps VERSION to v2.1.0
  - Closes issue with version report
```

#### 4. Upward Resolution

Each layer unblocks sequentially:

```
Layer 3 closes → Layer 4 unblocks
Layer 4 closes → Layer 5 unblocks
Layer 5 closes → User notified
```

#### 5. Validation

Foreman validates cross-layer contracts:

```bash
python foreman/test_observation_chain.py  # Must pass ✅
```

---

## Running a Cascade

### As the Initiating Layer (Layer N)

**When you discover the issue:**

1. **Analyze scope**: Can I fix this entirely within my layer?
   - ✅ Yes → Implement and close
   - ❌ No → Prepare fix-request

2. **Identify dependencies**: What do I need from Layer N-1?
   - Be specific: "I need a KinematicState message"
   - Include context: "For terrain estimation in Layer 4"
   - If you know it needs N-2, say so: "This may require Layer 3 changes"

3. **File fix-request**:
   ```bash
   # Execute fix-request workflow
   # Reference: ../philosophy/workflows/fix-request.json
   # Create GitHub issue in Layer N-1's repo
   ```

4. **Mark yourself BLOCKED**: Don't continue work until N-1 resolves

5. **Track in foreman**: Notify foreman to create cascade tracking file

### As an Intermediate Layer (Layer N-1)

**When you receive a fix-request:**

1. **Execute resolve-issues**:
   ```bash
   # Reference: ../philosophy/workflows/resolve-issues.json
   ```

2. **Analyze scope**: Can I fix this entirely within my layer?
   - ✅ Yes → Implement (proceed to step 5)
   - ❌ No → Need to cascade further (proceed to step 3)

3. **If cascading further**:
   - File fix-request to Layer N-2
   - Mark yourself BLOCKED
   - Include context from Layer N's request
   - Let Layer N-2 know if deeper layers might be involved

4. **Wait** (blocked) until Layer N-2 closes their issue

5. **When unblocked**:
   - Verify Layer N-2's fix works
   - Implement your layer's changes
   - All tests must pass
   - Commit, push, bump VERSION
   - Close issue with version report

### As the Resolving Layer (Bottom of Cascade)

**When you receive a fix-request and can resolve:**

1. **Implement the fix** entirely within your scope

2. **Validate**:
   - All unit tests pass (`make test`)
   - No boundary violations
   - No scope creep

3. **Commit and push**:
   ```bash
   git add <files>
   git commit -m "Fix: <description>

   Closes #<issue-number>

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
   git push origin main
   ```

4. **Bump VERSION**: Follow semantic versioning
   - MAJOR: Breaking changes to API
   - MINOR: New features, API additions
   - PATCH: Bug fixes

5. **Close issue** with version report:
   ```markdown
   Fixed in v2.1.0 (commit abc123f)

   **Changes**:
   - Added KinematicState publisher
   - Published at 100 Hz on rt/kinematic_state topic

   **Verification**:
   Layer 4 can now subscribe to rt/kinematic_state and receive:
   - foot_positions_world (4×3 array, meters)
   - foot_contact_states (4-bool array)
   - See layer_3/docs/api.md for full message spec
   ```

### As Foreman (Tracking)

**When a cascade initiates:**

1. **Create tracking file**:
   ```bash
   cp foreman/cascades/template.md \
      foreman/cascades/active/2026-XX-description.md
   ```

2. **Fill in details**:
   - Initiating layer and issue link
   - Expected cascade depth
   - What's being requested

3. **Monitor progress**:
   - Update as fix-requests are filed
   - Track blocking relationships
   - Note version bumps

4. **When cascade completes**:
   - Run integration tests
   - Verify all layers closed issues
   - Move to `completed/` or `failed/`
   - Notify user

**Foreman does NOT:**
- ❌ File fix-requests on behalf of layers
- ❌ Implement fixes
- ❌ Rush layers to close issues faster

---

## Real Example: Terrain-Aware Locomotion

See `foreman/cascades/completed/2026-02-terrain-aware-locomotion.md` for a complete cascade that:

- Initiated at Layer 5 (gait selection)
- Cascaded to Layer 4 (terrain estimation)
- Resolved at Layer 3 (kinematic state)
- Flowed back up with each layer completing
- Integration tests validated ✅

**Key insight**: Layer 4 issue #18 (HNN learning) was **rejected** during the cascade because learning violated Layer 4's stateless boundary. The cascade pattern **caught this architectural violation** and redirected to the correct approach (rule-based estimation).

---

## Anti-Patterns (Don't Do This)

### ❌ Layer Skipping

**Wrong**:
```
Layer 5 files fix-request directly to Layer 3
(skips Layer 4)
```

**Why wrong**: Breaks sovereignty, Layer 4 doesn't know about changes that affect it

**Right**:
```
Layer 5 → Layer 4 → Layer 3
(preserves hierarchy)
```

### ❌ Cross-Layer Editing

**Wrong**:
```python
# In layer_5/ repo
import sys
sys.path.append("../layer_3")
from layer_3.ik_solver import solve_ik

# Modify Layer 3's code directly
solve_ik.add_new_parameter(...)
```

**Why wrong**: Violates sovereignty, Layer 3 doesn't know about changes

**Right**:
```
File fix-request to Layer 4 (your N-1)
Let Layer 4 file fix-request to Layer 3
Wait for cascade to resolve
```

### ❌ Premature Resolution

**Wrong**:
```
Layer 4 closes issue while Layer 3 is still implementing
```

**Why wrong**: Breaks blocking discipline, Layer 4 might build on incomplete work

**Right**:
```
Layer 4 waits (BLOCKED) until Layer 3 closes
Then Layer 4 verifies and completes
```

### ❌ Scope Creep During Cascade

**Wrong**:
```
Layer 3 receives fix-request for KinematicState
Layer 3 also adds DynamicState, BehavioralState, logging, profiling...
```

**Why wrong**: Adds features no one asked for, delays cascade

**Right**:
```
Layer 3 implements exactly what was requested
If other improvements are needed, file separate issues
```

---

## Cascade Checklist

### Before Filing Fix-Request

- [ ] I've analyzed scope (can't fix entirely within my layer)
- [ ] I've identified specific needs from Layer N-1
- [ ] I've checked Layer N-1's API docs (maybe it already exists?)
- [ ] I've documented why I need this (not just what)
- [ ] I'm filing to Layer N-1 only (not skipping layers)

### Before Closing Issue

- [ ] All changes are within my layer's scope
- [ ] All unit tests pass (`make test`)
- [ ] Code committed and pushed
- [ ] VERSION bumped appropriately
- [ ] Closing comment includes version report
- [ ] Verified upstream layer can use the changes

### Foreman Validation

- [ ] Integration tests pass
- [ ] All layers in cascade closed issues
- [ ] No sovereignty violations occurred
- [ ] Cascade tracking file updated
- [ ] User notified of completion

---

## Success Metrics

A well-executed cascade should have:

✅ **Clear boundaries** — Each layer knew what was in scope
✅ **Proper blocking** — No layer proceeded before dependencies resolved
✅ **Version coordination** — Each layer bumped VERSION appropriately
✅ **Integration validation** — Tests pass across all affected layers
✅ **Traceability** — GitHub issues and cascade tracking show full history

---

## See Also

- `philosophy/workflows/cascade.json` — Formal workflow definition
- `philosophy/workflows/fix-request.json` — How to file fix-requests
- `philosophy/workflows/resolve-issues.json` — How to resolve issues
- `philosophy/boundaries.md` — Layer sovereignty and boundaries
- `philosophy/scope.md` — Information economy and need-to-know
- `foreman/cascades/README.md` — Tracking system documentation

---

**Last updated**: 2026-02-11
**Status**: Active pattern
**Proven**: 1 successful cascade (terrain-aware locomotion)
