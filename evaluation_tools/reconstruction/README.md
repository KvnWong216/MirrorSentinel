# Reconstruction Priors for Mirror/Glass SLAM

This folder contains local wrappers that connect video reconstruction methods to
the ROS2 Sentinel-LIO evaluation pipeline.

The current positioning is:

- NeuralRecon: upstream reconstruction prior for our method.
- SLAM3R / MASt3R-SLAM: reconstruction baselines for comparison.
- `/vfm/mirror_mask`: optional risk prior only; it should not be the core method.

Third-party code is kept under:

```text
third_party/reconstruction/
  NeuralRecon/
  SLAM3R/
  MASt3R-SLAM/
```

## NeuralRecon

NeuralRecon's official demo expects:

```text
DATAROOT/
  images/0.jpg
  images/1.jpg
  ...
  fragments.pkl
  SyncedPoses.txt
```

`prepare_neuralrecon_demo.py` creates this format from a ROS2 bag and a TUM
trajectory.  The trajectory should be exported from a Sentinel-LIO run, usually
`fast_lio2_equiv` for a neutral pose source.

### 1. Prepare Environment

```bash
bash evaluation_tools/reconstruction/setup_neuralrecon_env.sh
```

On this workstation the environment is:

```text
neuralrecon-sentinel
```

If conda cannot install a full CUDA toolkit, build the local TorchSparse
extension with the fallback CUDA headers already cached by conda:

```bash
bash evaluation_tools/reconstruction/build_torchsparse_local.sh
```

This writes:

```text
third_party/reconstruction/torchsparse/torchsparse/backend.cpython-310-x86_64-linux-gnu.so
```

Download official ScanNet pretrained weights into:

```text
third_party/reconstruction/NeuralRecon/checkpoints/
```

The helper command is:

```bash
bash evaluation_tools/reconstruction/download_neuralrecon_checkpoint.sh
```

The checkpoint filename should end with `.ckpt`, because `demo.py` loads the
latest `*.ckpt` under `LOGDIR`.

### 2. Get a Pose Source

If a Sentinel-LIO result already exists, reuse its trajectory:

```text
evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/trajectory_tum.txt
```

Otherwise, first run the neutral baseline:

```bash
python3 evaluation_tools/run_ros2_slam_eval.py \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --sequence 2026-03-30-21-31-03_fastlio2_equiv_v3_sparse \
  --method fast_lio2_equiv \
  --timeout 180
```

### 3. Convert ROS2 Bag to NeuralRecon Demo Data

```bash
python3 evaluation_tools/reconstruction/prepare_neuralrecon_demo.py \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --trajectory-tum evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/trajectory_tum.txt \
  --out-dir dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued \
  --every 3 \
  --n-views 9 \
  --fragment-stride 9 \
  --max-pose-dt 0.08
```

Outputs:

```text
dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued/
  images/
  fragments.pkl
  SyncedPoses.txt
  timestamps.txt
  image_metadata.json
  conversion_metadata.json
```

The script reads camera intrinsics and LiDAR/camera extrinsics from:

```text
Sentinal-LIO_ws/src/sentinel_lio/config/sentinel_ouster.yaml
```

It assumes `sentinel.extrinsic_R/t` follows the current Sentinel projection
convention, `p_camera = T_body_to_camera * p_body`.  NeuralRecon expects
camera-to-world poses, so the converter inverts this transform before writing
`fragments.pkl`.  If inspection shows the reconstructed mesh is flipped or
badly displaced, regenerate with:

```bash
--extrinsic-direction camera_to_body
```

### 4. Run NeuralRecon

```bash
bash evaluation_tools/reconstruction/run_neuralrecon_rescued.sh
```

The wrapper writes:

```text
dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued/neuralrecon_demo.yaml
```

and then runs:

```text
third_party/reconstruction/NeuralRecon/demo.py
```

NeuralRecon saves meshes under its official `results/` directory.

## Baselines

### SLAM3R

Official repo:

```text
third_party/reconstruction/SLAM3R
```

Install following its README.  It supports folder/video input and online/offline
reconstruction.  Use the same extracted image folder for fair comparison:

```bash
python evaluation_tools/reconstruction/run_slam3r_baseline.py \
  --dataset dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued/images \
  --test-name 2026-03-30-21-31-03_rescued_slam3r \
  --online
```

### MASt3R-SLAM

Official repo:

```text
third_party/reconstruction/MASt3R-SLAM
```

Install with submodules and checkpoints according to its README.  It can run on
an image folder.  The wrapper can generate `mast3r_intrinsics.yaml` from the
NeuralRecon conversion metadata:

```bash
python evaluation_tools/reconstruction/run_mast3r_slam_baseline.py \
  --dataset dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued/images \
  --metadata dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued/conversion_metadata.json
```

To run without calibration:

```bash
python evaluation_tools/reconstruction/run_mast3r_slam_baseline.py \
  --dataset dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued/images \
  --no-calib
```

## Next Integration Step

NeuralRecon currently produces a mesh/TSDF-style reconstruction prior.  The SLAM
side still needs a map-level reconstruction consistency gate:

```text
LiDAR map point -> nearest NeuralRecon surface / TSDF residual
consistent observations -> keep
behind/reconstruction-inconsistent observations -> accumulate ghost score
clean map export -> remove high ghost-score voxels
```

This keeps LIO pose estimation stable while making the final map reject mirror
and glass ghost geometry.
