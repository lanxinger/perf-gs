# PERF-GS: 3DGS in 10 seconds

**Efficient, extensible, dependency-free, low-memory 3D Gaussian Splatting training in Python and Slang.**

[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://electronicarts.github.io/perf-gs/)
[![Platforms: Windows and macOS](https://img.shields.io/badge/platforms-Windows%20%2B%20macOS-4A6B82)](#requirements)
[![Languages: Python and Slang](https://img.shields.io/badge/languages-Python%20%2B%20Slang-CF7434)](#architecture-from-a-high-level)
[![Backends: Vulkan, D3D12 and Metal](https://img.shields.io/badge/backends-Vulkan%20%2B%20D3D12%20%2B%20Metal-355B3E)](#requirements)

Open a folder of photos (with built-in COLMAP reconstruction) or an existing COLMAP scene in the standalone app, and train a 3DGS model with Slang on Vulkan, D3D12, or native Metal (Apple Silicon), no CUDA or PyTorch required.<br>
Across the nine Mip-NeRF 360 scenes on an **RTX 4090**, PERF-GS averages:
- **27.13 dB in 10.51 s** (*Ultrafast*)
- **27.40 dB in 15.91 s** (*Balanced*)
- **27.90 dB in 22.68 s** (*Quality*)
- **28.23 dB in 122.66 s** (*Quality+*).

To the best of our knowledge (as of November 2026), PERF-GS is the fastest open-source 3DGS trainer, outperforming [LiteGS](https://github.com/MooreThreads/LiteGS), [SAD-GS](https://github.com/LinjieLyu/SADGS), and [FastGS](https://github.com/fastgs/FastGS) in training speed with competitive or higher reconstruction quality and low VRAM usage (down to **2.80 GiB** with *Balanced* on a 6 GB RTX 4050 Laptop GPU).

[Quickstart](#quickstart) | [Benchmarks](#benchmarks) | [Reconstructions](#reconstructions) | [Design](#architecture-from-a-high-level)

<div align="center">

![The Stump training timer reaches 9.21 seconds at step 3,000, then the camera orbits the reconstruction.](assets/stump-demo.webp)

*Stump - Ultrafast - RTX 4090 - 9.21 s training loop (13.08 s from scene open)* | [GIF](assets/stump-preview.gif) | [Full-res WebP](assets/stump-rtx4090-orbit.webp)

</div>

## Requirements

- **OS & Python:** Python 3.10+. Tested on Windows 11 and macOS (Apple Silicon via native Metal, or Vulkan/MoltenVK). Linux is not yet verified.
- **GPU:** Tested on NVIDIA RTX 4050 Laptop (6 GB), RTX 4080 (16 GB), RTX 4090 (24 GB), AMD Radeon RX 7900 XT (20 GB), and Apple Silicon (M4 Pro native Metal; M2 via MoltenVK). Uses native FP32 `atomic_float` gradients on Metal, 64-bit integer atomics when supported (Vulkan/D3D12), and falls back to 32-bit fixed-point accumulators on Vulkan devices without int64 atomics (e.g., MoltenVK).
- **VRAM:** *Balanced* uses ~2.8 GiB on Mip-NeRF 360 (Windows/Vulkan) and 2.19 GiB base GPU buffers (2.23 GiB total allocated) on Apple Silicon Metal by streaming photos from system RAM. [Memory breakdown](docs/benchmarks.md#where-memory-goes)

## Quickstart

### Windows (GUI Hub)

1. **Get the code:** `git clone https://github.com/electronicarts/perf-gs.git` (or download a ZIP from **Releases** / **Code**).
2. **Create a Python 3.11 environment and install** (if `py -3.11 --version` fails, install Python 3.11 via the [Python install manager](https://www.python.org/downloads/windows/)):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e .
   ```

3. **Launch the hub:**

   ```powershell
   .\.venv\Scripts\python.exe -m trainer.frontend.app --hub
   ```

4. **Train & export:** Click **Open dataset folder...** (for raw photos, click **Install reconstruction tools** → **Reconstruct to COLMAP**), click **Start Training**, then **Export PLY...**.

<p align="center">
  <a href="assets/train_hub.png"><img src="assets/train_hub.png" width="49%" alt="Train scene in the hub after 3,000 steps, with the Export PLY button visible"></a>
  <a href="assets/room_hub.png"><img src="assets/room_hub.png" width="49%" alt="Room scene in the hub before training, with the Balanced preset selected"></a>
</p>

### macOS (Apple Silicon — Native Metal)

On Apple Silicon, `--backend metal` is selected automatically (no Homebrew or MoltenVK required) and unified memory budgeting queries `MTLDevice.recommendedMaxWorkingSetSize`. Ensure Apple's Metal Toolchain is installed (`xcodebuild -downloadComponent MetalToolchain` if `xcrun -sdk macosx metal --version` fails):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .

# Launch the interactive hub:
.venv/bin/python -m trainer.frontend.app --hub

# Or download Tanks & Temples and train headlessly:
.venv/bin/python scripts/data/download_tandt_db.py --set tandt
.venv/bin/python -m trainer.frontend.app --scene datasets/tandt/truck --no-viewer --save-ply
```

<details>
<summary><strong>Alternative: macOS via MoltenVK (<code>--backend vulkan</code>)</strong></summary>

If you prefer upstream's Vulkan-on-MoltenVK path (32-bit fixed-point accumulators):

```sh
brew install molten-vk vulkan-loader
xcodebuild -downloadComponent MetalToolchain
.venv/bin/python -m trainer.frontend.app --backend vulkan --hub
```

</details>

### Command Line

Download the Tanks & Temples Truck scene (~650 MB) and train headless with *Balanced* (default; add `--preset ultrafast|quality|quality+` to switch):

```powershell
.\.venv\Scripts\python.exe scripts/data/download_tandt_db.py --set tandt
.\.venv\Scripts\python.exe -m trainer.frontend.app --scene "datasets/tandt/truck" --no-viewer --save-ply
```

Outputs are written under `outputs/train/<timestamp>/output.ply`. Run `--help` or see [Usage](docs/usage.md#training) | [Hub walkthrough](docs/usage.md#interactive-hub) | [Datasets](docs/usage.md#datasets).

<a id="mip-nerf-360"></a>

## Benchmarks

### Mip-NeRF 360 Presets (RTX 4090)

| Preset | Steps | Train time | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Splats | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | 3,000 | 10.51 s | 27.13 dB | 0.7935 | 0.2564 | 1.43M | 5.29 GiB |
| Balanced *(default)* | 3,000 | 15.91 s | 27.40 dB | 0.8141 | 0.2228 | 2.07M | 2.81 GiB |
| **Quality** | **3,500** | **22.68 s** | **27.90 dB** | **0.8285** | **0.1996** | **2.10M** | **4.81 GiB** |
| **Quality+** | 17,044 avg. | 122.66 s | 28.23 dB | 0.8317 | 0.1909 | 2.10M | 4.81 GiB |

### Hardware Scaling (Mip-NeRF 360)

Quality metrics stay within **0.02 dB PSNR** (and **0.0005** SSIM / LPIPS) across all four tested GPUs; only training time and VRAM vary.

| GPU | Ultrafast | Balanced | Quality | Quality+ | Details |
| --- | ---: | ---: | ---: | ---: | --- |
| **NVIDIA RTX 4090** *(24 GB)* | 10.51 s *(5.29 GiB)* | 15.91 s *(2.81 GiB)* | 22.68 s *(4.81 GiB)* | 122.66 s *(4.81 GiB)* | [CSV](docs/NVIDIA_GeForce_RTX_4090_presets_current.csv) | [Q+](docs/NVIDIA_GeForce_RTX_4090_qualityplus.csv) | [Repro](docs/benchmark-rtx4090.md) |
| **NVIDIA RTX 4080** *(16 GB)* | 17.62 s *(5.29 GiB)* | 25.97 s *(2.80 GiB)* | 32.44 s *(4.80 GiB)* | 216.40 s *(4.80 GiB)* | [CSV](docs/NVIDIA_GeForce_RTX_4080_presets.csv) | [Q+](docs/NVIDIA_GeForce_RTX_4080_qualityplus.csv) |
| **AMD RX 7900 XT** *(20 GB)* | 16.44 s *(5.41 GiB)* | 27.02 s *(2.85 GiB)* | 39.47 s *(4.85 GiB)* | 202.04 s *(4.85 GiB)* | [CSV](docs/AMD_Radeon_RX_7900_XT_presets.csv) | [Q+](docs/AMD_Radeon_RX_7900_XT_qualityplus.csv) |
| **NVIDIA RTX 4050 Laptop** *(6 GB, 30 W)* | 90.50 s\* *(3.20 GiB)* | 139.88 s *(2.80 GiB)* | 191.41 s *(4.77 GiB)* | - | [CSV](docs/NVIDIA_GeForce_RTX_4050_Laptop_GPU_benchmarks.csv) | [Repro](docs/benchmark-rtx4050.md) |

*\*On the 6 GB RTX 4050 Laptop, Ultrafast streamed indoor images from RAM on Bonsai, Counter, Kitchen, and Room to fit the 4.49 GiB local VRAM budget (stock Ultrafast completed 5/9 scenes in 69.54 s avg.).*

### Speed vs. Quality Frontier (RTX 4080)

![Training-loop time and held-out PSNR for Gaussian splatting trainers measured on one RTX 4080.](docs/assets/mipnerf360-rtx4080-frontier.png)

Same-GPU comparison on all nine Mip-NeRF 360 scenes: PERF-GS *Quality* reaches **27.90 dB in 32.44 s** vs. LiteGS (**27.58 dB / 59.27 s**), SAD-GS (**27.19 dB / 159.21 s**), SkipGS (**27.43 dB / 184.26 s**), and FastGS-Big (**27.83 dB / 325.86 s**). On the 7-scene subset used by gsplat/LichtFeld (excluding *Flowers* and *Treehill*), *Quality+* averages **29.83 dB**. [Full trainer comparison & tables](docs/benchmarks.md#rtx-4080-comparison)

### Apple Silicon (Native Metal — Apple M4 Pro)

Measured on an **Apple M4 Pro** (24 GB unified memory, 17.76 GiB Metal working-set budget) on Tanks & Temples (`--backend metal --seed 42 --no-viewer --final-test-psnr`):

| Scene | Preset | Resolution | Steps | Train time | Throughput | Held-out PSNR ↑ | Splats | VRAM in use |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `truck` | Ultrafast (`--stbn-source white`) | 489×272 (4×) | 3,000 | **33.8 s** | **88.8 it/s** | 27.41 dB | 0.91M | 2.14 GiB |
| `truck` | Balanced *(default)* | 489×272 (4×) | 3,000 | **42.3 s** | **71.0 it/s** | 27.93 dB | 1.01M | 2.23 GiB |
| `truck` | **Quality** | 489×272 (4×) | 3,500 | **54.0 s** | **64.9 it/s** | 28.28 dB | 0.76M | 4.11 GiB |
| `truck` | **Quality+** | 489×272 (4×) | 29,780 | **361.9 s** | **82.3 it/s** | **28.78 dB** | 0.76M | 4.13 GiB |
| `truck` | Balanced | 978×545 (2×) | 3,000 | **89.7 s** | 33.5 it/s | 27.05 dB | 1.78M | 2.34 GiB |
| `truck` | Balanced | 1957×1091 (1×) | 3,000 | **189.5 s** | 15.8 it/s | 27.10 dB | 1.72M | 2.74 GiB |
| `train` | Balanced | 489×272 (4×) | 3,000 | **45.5 s** | 66.0 it/s | 21.81 dB | 1.06M | 2.23 GiB |

Native Metal uses packed 56 B `Gaussian3D` structs (layout parity with Vulkan/D3D12), wave-compacted visible-only `depth_sort` and `sh_forward`, hardware FP32 `atomic_float` gradients, Metal SIMD prefix-scan backward, cooperative 32-splat bucket preloading in forward rasterization, and sync-bracketed per-stage profiling via `--stage-timing-every`.

<a id="rtx-4050-laptop-gpu"></a>
<a id="rtx-4050-laptop-gpu-on-truck"></a>
<details>
<summary><strong>More benchmarks: Tanks &amp; Temples + Deep Blending (RTX 4050 Laptop, 6 GB / 30 W)</strong></summary>

| Dataset | Preset | Steps | Train time | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Splats | VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mip-NeRF 360 *(9 scenes)* | Ultrafast *(stream indoor)* | 3,000 | 90.50 s | 27.11 dB | 0.7935 | 0.2566 | 1.43M | 3.20 GiB |
| Mip-NeRF 360 *(9 scenes)* | Balanced | 3,000 | 139.88 s | 27.40 dB | 0.8143 | 0.2228 | 2.07M | 2.80 GiB |
| Mip-NeRF 360 *(9 scenes)* | Quality | 3,500 | 191.41 s | 27.89 dB | 0.8285 | 0.1996 | 2.10M | 4.77 GiB |
| Tanks & Temples *(2 scenes)* | Ultrafast | 3,000 | 73.91 s | 23.92 dB | 0.8379 | 0.1954 | 1.37M | 3.71 GiB |
| Tanks & Temples *(2 scenes)* | Balanced | 3,000 | 105.40 s | 24.10 dB | 0.8507 | 0.1748 | 1.73M | 2.60 GiB |
| Tanks & Temples *(2 scenes)* | Quality | 3,500 | 123.24 s | 24.47 dB | 0.8579 | 0.1660 | 1.30M | 4.60 GiB |
| Deep Blending *(2 scenes)* | Ultrafast | 3,000 | 51.04 s | 29.89 dB | 0.9036 | 0.2741 | 1.02M | 4.60 GiB |
| Deep Blending *(2 scenes)* | Balanced | 3,000 | 84.99 s | 29.91 dB | 0.9078 | 0.2585 | 1.70M | 2.73 GiB |
| Deep Blending *(2 scenes)* | Quality | 3,500 | 115.19 s | 30.19 dB | 0.9114 | 0.2458 | 1.83M | 4.72 GiB |

[All 39 measurements](docs/NVIDIA_GeForce_RTX_4050_Laptop_GPU_benchmarks.csv) | [System info](docs/NVIDIA_GeForce_RTX_4050_Laptop_GPU_system.txt) | [Reproduce](docs/benchmark-rtx4050.md) | [Truck telemetry](docs/benchmarks.md#truck-timing-and-memory)

</details>

<details>
<summary><strong>Per-scene benchmark charts (RTX 4090, RTX 4080, RX 7900 XT)</strong></summary>

![RTX 4090 per-scene presets](assets/mipnerf360-rtx4090-presets.png)

![RX 7900 XT per-scene presets](docs/assets/rx7900xt-presets.png)

![RTX 4080 per-scene presets](assets/mipnerf360-rtx4080-presets.png)

</details>

**Protocol:** Headless Vulkan, seed 42, equal scene weights, 1/8 camera holdout (246 Mip-NeRF 360 views at 1/4 outdoor / 1/2 indoor; 70 T&T at 1/2; 62 DB at full res), scored with Graphdeco RGB8 PSNR, SSIM, and VGG LPIPS. *Train time* covers the optimization loop through final GPU sync (excluding loading, evaluation, and export); *VRAM* is loop-end process-local GPU memory. [Full benchmark report, cross-vendor checks & methodology](docs/benchmarks.md) | [Reproduction protocol](docs/mipnerf360_3k.md#reproduce-the-benchmark)

<a id="architecture-from-a-high-level"></a>

## Overview

PERF-GS is built to run hundreds of experiments per day on a single consumer GPU - on an RTX 4090, *Ultrafast* completes all nine Mip-NeRF 360 scenes in 94.6 s of training time.

<details>
<summary><strong>Trainer design</strong></summary>

[Initialization](#initialization) | [Forward passes](#forward-passes) | [Backward passes](#backward-passes) | [Densification](#densification) | [Presets](#presets)

<a name="why-slangpy-and-slang"></a>

### Trainer architecture

We built on **SlangPy** to keep the training schedule modular in Python while writing fast, differentiable kernels in Slang - no separate native extension to build. SlangPy compiles and caches shader entry points directly from source and dispatches them through its RHI.

A *primitive* defines its projection, pixel coverage, and color; the *compositor* blends it. Slang.D differentiates projection and coverage directly from forward source, while outer backward shells are hand-written for speed. A small disk primitive in the test suite reuses the same forward blend loop. [Primitive interfaces](shaders/core/interfaces.slang) | [Disk primitive test](tests/test_disk_primitive.py)

Vulkan is the primary cross-platform target (including Apple Silicon via MoltenVK), and D3D12 on Windows shares the exact same shaders. Training has no CUDA or PyTorch dependency and supports both 32- and 64-lane waves across NVIDIA, AMD, and Apple GPUs.

<a name="where-the-first-splats-come-from"></a>

### Initialization

COLMAP sparse points initialize isotropic splats (opacity 0.1, scale from RMS distance to the 3 nearest neighbors). Up to 28 anchor training views are stereo-matched and three-view color-verified to seed up to 256,000 additional splats in under-sampled regions. For single-camera reconstructions, a radial lens-vignetting falloff strength is fitted once during setup (inspired by [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/)). [Stereo init](src/trainer/data/stereo.py) | [Lens fit](src/trainer/data/lens_calibration.py)

<a name="drawing-one-training-view"></a>

### Forward passes

Each step records forward render, image loss, backward passes, and Adam into one GPU submission. After growth, Morton reordering enables 128-splat cluster frustum culling. Splats are depth-sorted **once per view**, tile pairs are emitted in depth order, and a stable GPU radix sort orders only tile IDs. Rendering uses Mip-Splatting (3D + 2D filters with opacity compensation), skips $\alpha < 1/255$, and early-outs below $T < 10^{-4}$. Loss combines pixel error with separable 11x11 SSIM. [Rasterizer](src/trainer/backend/render/rasterizer.py) | [SSIM shader](shaders/losses/ssim_loss.slang)

<a name="the-backward-passes"></a>

### Backward passes

Inspired by [TamingGS](https://arxiv.org/abs/2406.15643), the forward pass checkpoints accumulated color and transmittance every 32 splats per tile. Indirect dispatches launch one wave per 32-splat bucket so each lane accumulates pixel gradients for one splat in registers using wave intrinsics before atomic reduction - drastically lowering memory contention. Separate passes differentiate projection and spherical harmonics once per contributing splat. Adam updates FP32 parameters from fixed-point atomic gradients (64-bit where supported, 32-bit fallback otherwise) using stochastically rounded BF16 moments. [Blend backward](shaders/rasterize/tile_binned/backward_persplat.slang) | [Projection backward](shaders/rasterize/tile_binned/projection_backward.slang) | [Adam precision](shaders/common/adam_precision.slang)

<a name="deciding-where-to-grow"></a>

### Densification

Growth combines [SAD-GS](https://vcai.mpi-inf.mpg.de/projects/SAD-GS/) anisotropic structure-aware splitting (every 500 steps, projecting splat axes against image structure tensors) with interleaved 100-step [AbsGS](https://arxiv.org/abs/2404.10484)-style gradient clone/split and low-opacity / low-blend-weight pruning. When proposed splits exceed capacity, expansions are ranked by gradient per child so whole families receive slots, strongest axis first. [Structure scoring](shaders/densify/score_structure_aware.slang) | [Split ranking](shaders/densify/rank_families.slang)

</details>

<details>
<summary><strong>Preset recipes</strong></summary>

<a name="presets"></a>

| Preset | Steps | SH | Pixel sampling | Photos | Growth cap | Post-growth trim |
| --- | ---: | ---: | --- | --- | ---: | --- |
| Ultrafast | 3,000 | 2 | 25% [STBN](docs/stbn_pixel_sampling.md) | GPU cache | 1.85M | - |
| Balanced *(default)* | 3,000 | 2 | 100% | Streamed from RAM | 2.60M | - |
| Quality | 3,500 | 3 | 100% | Streamed from RAM | 4.00M | Keep 75% by RGB removal cost at step 2,100 |
| Quality+ | Until plateau | 3 | 100% | Streamed from RAM | 4.00M | Same as Quality, then refines until training PSNR stalls (<0.05 dB over 3 checks) |

[All CLI flags and schedules](docs/usage.md#training)

</details>

## Reconstructions

Seed 42 held-out renders (*Ultrafast* & *Balanced*: 3,000 steps; *Quality*: 3,500 steps). [Benchmarks](#benchmarks)

<table>
  <thead>
    <tr>
      <th scope="col" width="25%">Reference photo</th>
      <th scope="col" width="25%">Ultrafast</th>
      <th scope="col" width="25%">Balanced</th>
      <th scope="col" width="25%">Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Kitchen - DSCF0912.JPG - Quality 34.99 dB</th></tr>
    <tr>
      <td><a href="docs/assets/reconstructions/kitchen-reference.webp"><img src="docs/assets/reconstructions/kitchen-reference.webp" width="216" alt="Kitchen reference photograph and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/kitchen-ultrafast.webp"><img src="docs/assets/reconstructions/kitchen-ultrafast.webp" width="216" alt="Kitchen Ultrafast render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/kitchen-balanced.webp"><img src="docs/assets/reconstructions/kitchen-balanced.webp" width="216" alt="Kitchen Balanced render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/kitchen-quality.webp"><img src="docs/assets/reconstructions/kitchen-quality.webp" width="216" alt="Kitchen Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Bicycle - _DSC8824.JPG - Quality 29.74 dB</th></tr>
    <tr>
      <td><a href="docs/assets/reconstructions/bicycle-reference.webp"><img src="docs/assets/reconstructions/bicycle-reference.webp" width="216" alt="Bicycle reference photograph and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/bicycle-ultrafast.webp"><img src="docs/assets/reconstructions/bicycle-ultrafast.webp" width="216" alt="Bicycle Ultrafast render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/bicycle-balanced.webp"><img src="docs/assets/reconstructions/bicycle-balanced.webp" width="216" alt="Bicycle Balanced render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/bicycle-quality.webp"><img src="docs/assets/reconstructions/bicycle-quality.webp" width="216" alt="Bicycle Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Stump - _DSC9277.JPG - Quality 30.65 dB</th></tr>
    <tr>
      <td><a href="docs/assets/reconstructions/stump-reference.webp"><img src="docs/assets/reconstructions/stump-reference.webp" width="216" alt="Stump reference photograph and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/stump-ultrafast.webp"><img src="docs/assets/reconstructions/stump-ultrafast.webp" width="216" alt="Stump Ultrafast render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/stump-balanced.webp"><img src="docs/assets/reconstructions/stump-balanced.webp" width="216" alt="Stump Balanced render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/stump-quality.webp"><img src="docs/assets/reconstructions/stump-quality.webp" width="216" alt="Stump Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Room - DSCF4763.JPG - Quality 36.30 dB</th></tr>
    <tr>
      <td><a href="docs/assets/reconstructions/room-reference.webp"><img src="docs/assets/reconstructions/room-reference.webp" width="216" alt="Room reference photograph and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/room-ultrafast.webp"><img src="docs/assets/reconstructions/room-ultrafast.webp" width="216" alt="Room Ultrafast render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/room-balanced.webp"><img src="docs/assets/reconstructions/room-balanced.webp" width="216" alt="Room Balanced render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/room-quality.webp"><img src="docs/assets/reconstructions/room-quality.webp" width="216" alt="Room Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Bonsai - DSCF5725.JPG - Quality 35.45 dB</th></tr>
    <tr>
      <td><a href="docs/assets/reconstructions/bonsai-reference.webp"><img src="docs/assets/reconstructions/bonsai-reference.webp" width="216" alt="Bonsai reference photograph and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/bonsai-ultrafast.webp"><img src="docs/assets/reconstructions/bonsai-ultrafast.webp" width="216" alt="Bonsai Ultrafast render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/bonsai-balanced.webp"><img src="docs/assets/reconstructions/bonsai-balanced.webp" width="216" alt="Bonsai Balanced render and detail crop"></a></td>
      <td><a href="docs/assets/reconstructions/bonsai-quality.webp"><img src="docs/assets/reconstructions/bonsai-quality.webp" width="216" alt="Bonsai Quality render and detail crop"></a></td>
    </tr>
  </tbody>
</table>

Scenes and reference photography from Barron et al., CVPR 2022. [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/)

## Datasets

The download helpers link to three datasets that can be obtained separately by users - [Download instructions](docs/usage.md#datasets)

| Dataset | Project / attribution | Included scene selection |
| --- | --- | --- |
| Mip-NeRF 360 | [Barron et al., CVPR 2022](https://jonbarron.info/mipnerf360/) | bicycle, bonsai, counter, garden, kitchen, room, stump (plus optional flowers and treehill) |
| Tanks & Temples | [Knapitsch et al., ACM TOG 2017](https://www.tanksandtemples.org/) | train, truck |
| Deep Blending | [Hedman et al., ACM TOG 2018](https://repo-sam.inria.fr/fungraph/deep-blending/) | drjohnson, playroom |

## License

This project is provided under the BSD 3-Clause License, see [LICENSE](LICENSE) for details. See [NOTICE](NOTICE.md) for licensing details on other software and data.

## Contributing

Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor. You can sign [here](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*).

## How to Cite

Scolari, S. (2026). *perf-gs*. Electronic Arts. <https://github.com/electronicarts/perf-gs>

```bibtex
@software{scolari2026perfgs,
  author    = {Scolari, Stefano},
  title     = {{perf-gs}},
  year      = {2026},
  publisher = {Electronic Arts},
  url       = {https://github.com/electronicarts/perf-gs}
}
```

## SEED

<p align="center"><a href="https://seed.ea.com"><img src="assets/SEED.jpg" width="150px"></a><br>
<b>Search for Extraordinary Experiences Division (SEED) - Electronic Arts <br> http://seed.ea.com</b><br>
We are a cross-disciplinary team within EA Worldwide Studios.<br>
Our mission is to explore, build and help define the future of interactive entertainment.<br>
PERF-GS was created by <a href="https://stefanoscolapasta.github.io/nozbufferallowed/about.html">Stefano Scolari</a>.</p>
