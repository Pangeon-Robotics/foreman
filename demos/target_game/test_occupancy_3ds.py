"""3DS metrics: surface adherence, completeness, phantom rate.

Provides compute_3ds_v2 and compute_3ds_god for comparing TSDF surface
accuracy against ground-truth obstacle geometry.
Extracted from test_occupancy.py to keep files under 400 lines.
"""
from __future__ import annotations

import numpy as np

from .test_occupancy_gt import extract_tsdf_surface, sample_gt_surfaces


def compute_3ds_v2(tsdf, scene_xml_path: str,
                    adherence_tol: float = 0.03,
                    phantom_tol: float = 0.05,
                    sample_spacing: float = 0.01) -> dict:
    """Compute 3DS v2 metrics: adherence, completeness, phantom rate.

    Parameters
    ----------
    tsdf : TSDF
    scene_xml_path : str
        Path to the scene XML file containing obstacle definitions.
    adherence_tol : float
        Completeness tolerance (meters). Default 3cm.
    phantom_tol : float
        Phantom tolerance (meters). Default 5cm.
    sample_spacing : float
        GT surface sampling density (meters). Default 1cm.

    Returns
    -------
    dict with adherence_mm, completeness_pct, phantom_pct, counts.
    """
    from scipy.spatial import cKDTree

    tsdf_surface = extract_tsdf_surface(tsdf)
    gt_surface = sample_gt_surfaces(scene_xml_path, sample_spacing)

    n_tsdf = len(tsdf_surface)
    n_gt = len(gt_surface)
    n_converged = getattr(tsdf, 'n_converged', 0)

    if n_tsdf == 0 or n_gt == 0:
        return {
            "adherence_mm": float('inf') if n_gt > 0 else 0.0,
            "completeness_pct": 0.0,
            "phantom_pct": 100.0 if n_tsdf > 0 and n_gt == 0 else 0.0,
            "n_tsdf_surface": n_tsdf,
            "n_gt_surface": n_gt,
            "n_converged": n_converged,
        }

    gt_tree = cKDTree(gt_surface)
    tsdf_tree = cKDTree(tsdf_surface)

    # (a) Surface adherence: mean TSDF->GT distance
    tsdf_to_gt_dists, _ = gt_tree.query(tsdf_surface, k=1)
    adherence_mm = float(np.mean(tsdf_to_gt_dists) * 1000.0)

    # (b) Completeness: fraction of GT points with a TSDF detection nearby
    gt_to_tsdf_dists, _ = tsdf_tree.query(gt_surface, k=1)
    completeness_pct = float(
        np.sum(gt_to_tsdf_dists < adherence_tol) / n_gt * 100.0)

    # (c) Phantom rate: fraction of TSDF voxels far from any GT surface
    phantom_pct = float(
        np.sum(tsdf_to_gt_dists > phantom_tol) / n_tsdf * 100.0)

    return {
        "adherence_mm": adherence_mm,
        "completeness_pct": completeness_pct,
        "phantom_pct": phantom_pct,
        "n_tsdf_surface": n_tsdf,
        "n_gt_surface": n_gt,
        "n_converged": n_converged,
    }


def compute_3ds_god(tsdf, scene_xml_path: str,
                    max_dist: float = 0.5,
                    completeness_tol: float = 0.05,
                    sample_spacing: float = 0.01,
                    gt_z_range: tuple[float, float] | None = None) -> dict:
    """God-mode 3DS: squared-distance penalty with omniscient GT.

    Parameters
    ----------
    tsdf : TSDF
    scene_xml_path : str
    max_dist : float
        Distance cap in meters. Default 0.5m.
    completeness_tol : float
        Detection tolerance. Default 5cm.
    sample_spacing : float
        GT surface sampling density. Default 1cm.
    gt_z_range : tuple[float, float], optional
        Filter GT surface points to this Z range for completeness scoring.

    Returns
    -------
    dict with score, precision_score, completeness_pct, phantom_penalty, counts.
    """
    from scipy.spatial import cKDTree

    tsdf_surface = extract_tsdf_surface(tsdf)
    gt_surface = sample_gt_surfaces(scene_xml_path, sample_spacing)

    # Precision/phantom use full GT (unfiltered)
    gt_surface_full = gt_surface

    # Completeness uses Z-filtered GT
    if gt_z_range is not None and len(gt_surface) > 0:
        z_lo, z_hi = gt_z_range
        z_mask = (gt_surface[:, 2] >= z_lo) & (gt_surface[:, 2] <= z_hi)
        gt_surface_filtered = gt_surface[z_mask]
    else:
        gt_surface_filtered = gt_surface

    n_tsdf = len(tsdf_surface)
    n_gt = len(gt_surface_filtered)
    n_gt_full = len(gt_surface_full)

    if n_tsdf == 0 and n_gt == 0:
        return {"score": 100.0, "precision_score": 100.0,
                "completeness_pct": 100.0, "phantom_penalty": 0.0,
                "n_tsdf": 0, "n_gt": 0}
    if n_gt_full == 0:
        return {"score": 0.0, "precision_score": 0.0,
                "completeness_pct": 100.0, "phantom_penalty": 100.0,
                "n_tsdf": n_tsdf, "n_gt": 0}
    if n_tsdf == 0:
        return {"score": 0.0, "precision_score": 100.0,
                "completeness_pct": 0.0, "phantom_penalty": 0.0,
                "n_tsdf": 0, "n_gt": n_gt}

    gt_tree_full = cKDTree(gt_surface_full)
    tsdf_tree = cKDTree(tsdf_surface)

    # (a) Precision: squared distance to nearest GT
    tsdf_to_gt, _ = gt_tree_full.query(tsdf_surface, k=1)
    capped = np.minimum(tsdf_to_gt, max_dist)
    mean_sq = float(np.mean(capped ** 2))
    precision_score = max(0.0, 100.0 * (1.0 - mean_sq / (max_dist ** 2)))

    # (b) Completeness: fraction of Z-filtered GT with TSDF nearby
    if n_gt > 0:
        gt_filtered_tree_dists, _ = tsdf_tree.query(gt_surface_filtered, k=1)
        completeness_pct = float(
            np.sum(gt_filtered_tree_dists < completeness_tol) / n_gt * 100.0)
    else:
        completeness_pct = 0.0

    # (c) Phantom penalty: mean squared distance of phantom voxels
    phantom_mask = tsdf_to_gt > completeness_tol
    n_phantom = int(np.sum(phantom_mask))
    if n_phantom > 0:
        phantom_dists = np.minimum(tsdf_to_gt[phantom_mask], max_dist)
        phantom_penalty = float(
            np.mean(phantom_dists ** 2) / (max_dist ** 2) * 100.0)
    else:
        phantom_penalty = 0.0

    # Composite: 40% precision + 20% completeness + 40% purity
    # Precision: are detected surfaces in the right place?
    # Purity: no phantom hallucinations?
    # Completeness: coverage (exploration-limited, weighted lower)
    purity_score = max(0.0, 100.0 - phantom_penalty)
    score = (0.40 * precision_score
             + 0.20 * completeness_pct
             + 0.40 * purity_score)
    score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 1),
        "precision_score": round(precision_score, 1),
        "completeness_pct": round(completeness_pct, 1),
        "phantom_penalty": round(phantom_penalty, 1),
        "n_tsdf": n_tsdf,
        "n_gt": n_gt,
    }
