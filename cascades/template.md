# Cascade: [SHORT-DESCRIPTION]

**Cascade ID**: `YYYY-MM-short-description`
**Status**: `initiated | cascading | resolving | complete | failed`
**Created**: YYYY-MM-DD
**Completed**: YYYY-MM-DD (or "in progress")

---

## Overview

**Initiated by**: User | Layer N
**Initiating layer**: Layer N
**Original issue**: Brief description of what user requested

---

## Cascade Flow

### Layer [N] (Initiating)

- **GitHub Issue**: [org/repo#issue](link)
- **Status**: `blocked | resolving | complete`
- **Blocked by**: Layer N-1 issue (if blocked)
- **Description**: What Layer N needs to do
- **Requires from N-1**: What Layer N needs from the layer below
- **Version**: vX.Y.Z (when resolved)
- **Closed**: YYYY-MM-DD (when resolved)

### Layer [N-1] (Intermediate)

- **GitHub Issue**: [org/repo#issue](link)
- **Status**: `blocked | resolving | complete`
- **Blocked by**: Layer N-2 issue (if blocked)
- **Description**: What Layer N-1 needs to do
- **Requires from N-2**: What Layer N-1 needs from the layer below (if applicable)
- **Version**: vX.Y.Z (when resolved)
- **Closed**: YYYY-MM-DD (when resolved)

### Layer [N-2] (Resolving)

- **GitHub Issue**: [org/repo#issue](link)
- **Status**: `resolving | complete`
- **Description**: What Layer N-2 needs to do (the actual fix)
- **Scope**: Entirely within Layer N-2's scope
- **Version**: vX.Y.Z (when resolved)
- **Closed**: YYYY-MM-DD (when resolved)

---

## Resolution Timeline

### Downward Cascade

- [ ] YYYY-MM-DD: Layer N files fix-request to Layer N-1
- [ ] YYYY-MM-DD: Layer N-1 files fix-request to Layer N-2
- [ ] YYYY-MM-DD: Layer N-2 begins implementation

### Upward Resolution

- [ ] YYYY-MM-DD: Layer N-2 completes, closes issue (vX.Y.Z)
- [ ] YYYY-MM-DD: Layer N-1 unblocks, completes, closes issue (vX.Y.Z)
- [ ] YYYY-MM-DD: Layer N unblocks, completes, closes issue (vX.Y.Z)

### Validation

- [ ] YYYY-MM-DD: Integration tests pass (`test_observation_chain.py`)
- [ ] YYYY-MM-DD: User notified of completion

---

## Integration Tests

**Tests to validate**:
- [ ] `python foreman/test_observation_chain.py`
- [ ] (Add cascade-specific tests if needed)

**Results**:
- Initial run: (pass/fail, date)
- Final run: (pass/fail, date)

---

## Notes

### Blockers Encountered
(Document any unexpected issues or delays)

### Key Insights
(What did we learn from this cascade?)

### Follow-up Work
(Any additional work identified during the cascade)

---

## Cascade Summary

**Total layers involved**: N
**Duration**: X days (from initiation to completion)
**Versions bumped**:
- Layer N: vX.Y.Z → vX.Y.Z
- Layer N-1: vX.Y.Z → vX.Y.Z
- Layer N-2: vX.Y.Z → vX.Y.Z

**Outcome**: Success ✅ | Failed ❌

---

**Tracked by**: Foreman
**Last updated**: YYYY-MM-DD
