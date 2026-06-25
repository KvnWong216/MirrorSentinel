# Reflection-Aware SLAM Stitching Protocol

This project is positioned as a ROS2 SLAM system that stitches existing visual
priors into LiDAR-inertial mapping.  The goal is not to invent a new foundation
model, but to make mirror/glass priors usable for real-time SLAM map cleaning.

## Reused Upstream Priors

- Reflection ROI / mask: 3DRef-style RGB reflection masks, SATNet/EBLNet outputs,
  or an oracle directory mask for controlled self-collected experiments.
- Depth prior: FastStereoFoundation / DA3-style depth publisher on
  `/vfm/depth_image`.
- Optional future priors: Mirror3D-style mirror plane/depth refinement or
  ClearGrasp/TransCG-style transparent-object depth completion.

## SLAM-Side Contribution

1. Consume upstream priors as ROS2 topics:
   - `/vfm/mirror_mask`
   - `/vfm/depth_image`
2. Use non-reflective projected LiDAR points as an online metric-depth
   calibration set:
   - estimate robust scale for visual depth using non-mask points
   - report raw and calibrated residuals in `/mirror_sentinel/frame_stats`
3. Apply depth residuals only inside reflection-risk ROI:
   - outside ROI: keep points and use them for calibration
   - inside ROI, LiDAR behind calibrated visual depth: reject/low-confidence ghost
   - inside ROI, LiDAR in front of visual depth: keep foreground object
   - invalid depth: uncertain; do not hard reject by default
4. Evaluate in SLAM terms:
   - behind-plane ghost point count
   - ghost rate in reflective ROI
   - reflective plane thickness
   - normal wall thickness
   - no-GT trajectory sanity metrics

## Current Method Names

- `fast_lio2_equiv`: no visual prior, baseline.
- `sentinel_rt_depth`: global depth prior, depth-only ablation.
- `sentinel_rt_depth_soft`: global depth soft weighting only.
- `sentinel_no_depth`: mask-only ablation.
- `sentinel_full`: reflection mask + non-mask depth calibration + ROI-only
  depth-confirmed map gating.

## Immediate Validation Order

1. Run `fast_lio2_equiv`.
2. Run `sentinel_full` with the self-collected oracle/draft mask prior.
3. Run `sentinel_rt_depth` to show why global depth is not enough.
4. Replace oracle mask with 3DRef-trained or SATNet/EBLNet masks.
5. Refine annotation plane/ROI in CloudCompare or RViz before reporting final
   table numbers.
