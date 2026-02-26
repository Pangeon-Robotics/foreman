# Changelog

All notable changes to foreman are documented in this file.

## 0.2.2 — 2026-02-26
- Add CHANGELOG.md with historical entries for all versions
- References: #3
Tests: **1/1 target reached (headless, SLAM, seed=42)**

## 0.2.1 — 2026-02-26
- Remove duplicate `normalize_angle` from `target.py`; import from canonical `utils.py` instead
- References: #2
Tests: **1/1 target reached (headless, SLAM, seed=42)**

## 0.2.0 — 2026-02-14
- Add VERSION file and `get_version()` API per engineering.md compliance
- Split `test_observation_chain.py` (579 lines) into `observation_chain.py` + `test_observation_chain.py` per 400-line limit
- Add target game demo with genome loading, DDS domain support, full-circle spawning
- Add cross-repo shared utilities in `utils.py` (quat_to_yaw, normalize_angle, frame_transform, etc.)
- Add Go2/Go2W/B2W support with differential wheel drive
- Integrate Layer 6 spatial awareness (SLAM, costmap, DWA) into target game
- Add obstacle avoidance scenario testing with automated critics
- Add DWA pipeline with heading-proportional control and A* path visualization
- Add ATO path critic, per-line telemetry, and progress-based early timeout

## 0.1.0 — 2026-02-11
- Initial commit: workspace coordination, CLAUDE.md, cascade pattern
- Add agent delegation pattern for layer sovereignty
- Establish foreman identity and boundaries
