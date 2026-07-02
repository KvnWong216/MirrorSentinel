# ROBIO 冲刺 TODO

目标很明确：MirrorSentinel 要证明的是 ROS2 Humble 在线 SLAM 中对镜子/玻璃鬼影点的稳定拒绝能力，而不是单纯做反射分割。当前仓库已经能完成 full bag 回放、建图、点云导出、人工房间边界评估和 oracle 上限评估；官方 DA3 metric depth 已经接入 ROS2 并跑出 depth-only 主候选结果。

## 0. 当前状态

- [x] ROS2 Humble 工程已可构建：`scripts/build_ros2.sh`
- [x] 完整自采 ROS1 bag 已转换为 ROS2 bag：`dataset/rosbag2/2026-03-30-21-31-03`
- [x] 手工房间 cuboid / 镜面墙边界标注已存在：`evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml`
- [x] 手工几何 reference map 已存在：`evaluation_tools/data/gt_reference/self_collected/2026-03-30-21-31-03_rescued/room_gt_reference.pcd`
- [x] FAST-LIO2-equivalent baseline 已跑通并评估
- [x] Manual geometry clean oracle 已跑通并评估
- [x] Heuristic `/vfm/depth_image` + `/vfm/mirror_mask` smoke test 已跑通
- [x] 真实 DA3 depth prior 已部署并跑通 ROS2 `/vfm/depth_image`
- [ ] 真实 reflection/mirror mask prior 资产缺失
- [x] DA3 depth-only `sentinel_rt_depth` 主候选结果已跑出
- [ ] `sentinel_full` depth+mask 完整变体结果缺失，但不再阻塞 depth-only 主线

每日刷新状态：

```bash
python3 evaluation_tools/mirrorsentinel_sprint_report.py
```

输出目录：

```text
evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_research_sprint/
```

## 1. 三天出结果

### Day 1：锁主结果来源

- [x] 找回或重新生成真实 `/vfm/depth_image`，不能再用 heuristic 当主方法。
- [ ] `/vfm/mirror_mask` 作为可选增强，不再阻塞第一版主实验。
- [ ] 如果找到历史 prior bag，用 `--prior-bag` 跑 `sentinel_full`。
- [ ] 如果只有模型权重/engine，跑实时节点生成 prior 并记录 outputs_bag。
- [ ] 产出 `sentinel_full` 的三份图：
  - `map_raw.pcd`
  - `map_marker_clean.pcd`
  - `map_vote_clean.pcd`
- [ ] 产出主方法指标：
  - `metrics_mapping_ablation.{json,csv,md}`
  - `metrics_runtime.{json,csv}`
  - `metrics_no_gt.json`
  - `metrics_reference_map_vote_clean.json`

主方法命令模板：

```bash
python3 evaluation_tools/run_ros2_slam_eval.py \
  --bag dataset/rosbag2/2026-03-30-21-31-03 \
  --prior-bag <REAL_VFM_OUTPUTS_BAG> \
  --sequence 2026-03-30-21-31-03_fullbag_mirrorsentinel_real_prior \
  --method sentinel_full \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --play-rate 1.0 \
  --startup-wait 5 \
  --shutdown-wait 10 \
  --timeout 180 \
  --frame-prior-post-clean \
  --frame-prior-frame-stride 5 \
  --frame-prior-max-pair-dt 0.30 \
  --frame-prior-selection-mode annotation_score_budget \
  --frame-prior-target-kept-points 100000 \
  --frame-prior-score-mode ghost_votes_ratio \
  --frame-prior-annotation-candidate-mode behind \
  --frame-prior-annotation-budget-weight candidate_count \
  --frame-prior-min-observations 1 \
  --frame-prior-min-ghost-votes 1 \
  --frame-prior-min-ghost-ratio 0.0
```

### Day 2：补消融

- [ ] `fast_lio2_equiv`：同代码公平 baseline。
- [ ] `sentinel_no_mask`：只用 depth consistency。
- [ ] `sentinel_no_depth`：只用 reflection mask。
- [ ] `sentinel_hard_reject`：硬剔除策略。
- [ ] `sentinel_full`：完整 soft weighting + backend historical cleanup。
- [ ] 所有方法必须使用同一 bag、同一 annotation、同一 reference map。

论文主表只放 4 到 6 个指标：

```text
Method | RER ↓ | Removal vs Raw ↑ | Valid Precision Proxy ↑ | Thickness P95 ↓ | Map Retention ↑ | FPS ↑
```

### Day 3：补泛化和图

- [ ] 至少补 1 条额外自采 mirror/glass ROS2 bag，优先电梯外/走廊玻璃。
- [ ] 或者补 3DRef 上游 prior 辅助表：IoU、F1、Precision、Recall、FPS。
- [ ] 生成图：
  - baseline vs ours top-view map
  - reflective ROI ghost focus
  - residual distance histogram
  - ablation bar chart
  - RViz pipeline screenshot

## 2. 一周写完论文

### Day 1：定题和协议

- [ ] 题目暂定：MirrorSentinel: Visual-Prior Assisted LiDAR-Inertial Mapping for Mirror/Glass Ghost Suppression。
- [ ] 明确本文不是 segmentation paper，而是 reflection-aware SLAM mapping paper。
- [ ] 固定评估协议：`SLAM_EVAL_PROTOCOL_CN.md`。

### Day 2：方法

- [ ] 写系统图：LiDAR/IMU + image prior + point projection + consistency voting + map cleanup。
- [ ] 写关键公式：
  - LiDAR 点投影到相机。
  - LiDAR range 与 visual depth 的残差。
  - mask/depth validity 到 point confidence。
  - historical votes 到 `map_vote_clean.pcd`。

### Day 3：实验设置

- [ ] 写传感器、ROS2 Humble、Ouster/ZED、外参、bag 时长、topic。
- [ ] 写人工 cuboid reference 的可信边界：manual geometric reference，不是 scanner-grade GT。

### Day 4：主结果和消融

- [ ] 主表：baseline/main/oracle/smoke test 分清角色。
- [ ] 消融表：no-mask/no-depth/hard/full。
- [ ] runtime 表：pipeline FPS、depth/mask prior FPS。

### Day 5：相关工作

- [ ] Non-Lambertian / mirror / glass SLAM。
- [ ] Reflection point removal / 3DRef。
- [ ] Foundation depth / video reconstruction prior。
- [ ] SLAM map quality evaluation：MapEval、ETH3D/Tanks-and-Temples 风格 cloud-to-reference。

### Day 6：图和讨论

- [ ] 定性图必须展示 SLAM 地图鬼影减少，而不是只展示 RGB mask。
- [ ] 失败案例：外参误差、prior 漏检、强反射、过度清理。
- [ ] 局限性：当前自采 reference 是 manual geometric reference。

### Day 7：定稿

- [ ] 摘要、贡献、结论统一措辞。
- [ ] 检查所有 claims 都有对应实验。
- [ ] README 一键复现实验命令更新。
- [ ] 大文件不进 GitHub。

## 3. 今晚最重要的判断

现在不能把 heuristic prior 结果写成主方法。它只证明 ROS2 topic 和 pipeline 运行正常。论文能不能成立，取决于三天内是否拿到真实 visual prior 下的 `sentinel_full` 结果，并且至少在 RER / Thickness P95 / FPS 上相对 `fast_lio2_equiv` 有可辩护提升。
