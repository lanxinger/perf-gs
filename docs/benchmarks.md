# Training benchmarks

## RTX 4050 Laptop GPU on Truck

The original timing check used Truck, the Tanks & Temples quickstart scene. Each preset trained once at seed 42 on 219 half-resolution images (978 × 545). Every eighth camera in filename order was held out, giving all three presets the same 32 test views. These first runs used the trainer's floating-point sRGB PSNR and SSIM. The full 13-scene comparison uses fresh Truck runs with saved renders and independent Graphdeco RGB8 scoring. [Three-dataset comparison](../README.md#rtx-4050-laptop-gpu) · [Full sweep protocol](benchmark-rtx4050.md) · [Original Truck measurements (CSV)](NVIDIA_GeForce_RTX_4050_Laptop_GPU_truck.csv)

The Dell XPS 14 9440 has a 6 GB RTX 4050 Laptop GPU. NVIDIA's driver reported a 30 W enforced power limit. The system used an Intel Core Ultra 7 155H, Windows 11 Enterprise (build 26100), NVIDIA driver 610.74, Python 3.13.2 and SlangPy 0.42.0 with PERF-GS at `26a4ca2`. Time covers the headless Vulkan training loop through final synchronization and excludes scene loading, initialization and held-out evaluation. VRAM measures the trainer process's local GPU memory at loop end.

To reproduce from the repository root after the Windows quickstart setup, fetch Truck and the NVIDIA sampling masks used by Ultrafast. Run each preset in its own process with the GPU idle. Ultrafast and Balanced default to 3,000 steps and Quality defaults to 3,500. [Windows setup](../README.md#quickstart) · [NVIDIA mask terms](stbn_pixel_sampling.md)

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python scripts/data/download_tandt_db.py --set tandt
& $python scripts/data/download_stbn.py
foreach ($preset in 'ultrafast', 'balanced', 'quality') {
    & $python -m trainer.frontend.app --scene datasets/tandt/truck --preset $preset `
        --backend vulkan --downsample 2 --seed 42 --no-viewer --evaluation `
        --final-test-psnr --log-every 0 --eval-every 0 --preview-every 0 `
        --no-write-ground-truths --run-report
}
```

Each run writes its settings, timings and per-view scores to `outputs/train/<run-name>/run_report.json`. The companion `test_psnr.json` records the loop-end VRAM usage in bytes.

### Truck timing and memory

The original Ultrafast run completed 3,000 steps in **73.11 s** (41.0 steps/s). The trainer recorded 73.06 s of active training and 0.05 s of in-loop observation. Its full process timer read 132.16 s after held-out evaluation. During the loop, two NVIDIA samples 30 s apart showed 100% GPU utilization at 29.94 W and 29.88 W, near the laptop's 30 W limit.

At loop end the trainer used 3.55 GiB of its 5.03 GiB local GPU memory budget. Its pipeline buffers accounted for 1.95 GiB, cached training photos for 0.52 GiB and per-view structure fields for 1.03 GiB. The run also held 1.50 GiB of decoded images in system RAM. [Truck measurements](NVIDIA_GeForce_RTX_4050_Laptop_GPU_truck.csv) · [How PERF-GS measures memory](#where-memory-goes)

An isolated Ultrafast repeat took **73.31 s**, within 0.3% of the original. Sampled dedicated GPU memory peaked at 3.56 GiB and shared GPU memory at 67 MiB. The laptop had over 36 GiB of system RAM available, and median PCIe traffic into the GPU was 4.8 MiB/s. The GPU ran at 100% median utilization and 29.94 W median power. The driver reported power-capped clocks in all 68 in-loop samples. This reproduces the timing with headroom in local VRAM and little evidence of sustained paging. [Truck repeat telemetry](NVIDIA_GeForce_RTX_4050_Laptop_GPU_telemetry.csv)

**Same-scene desktop comparison.** On Mip-NeRF 360 Stump, the 3,000-step Ultrafast loop took **62.41 s** on the RTX 4050 Laptop GPU, **13.17 s** on the RTX 4080 and **7.86 s** on the RTX 4090. All used seed 42 and 1,245 × 825 images. The archived RTX 4090 run used trainer revision `d69f4f2`. The laptop took 7.9× as long as the 4090. During the laptop run the trainer used at most 3.42 GiB of dedicated GPU memory and 67 MiB of shared GPU memory at the sample times. Median GPU utilization was 100% at 29.96 W. The driver reported power-capped clocks in 56 of 57 samples and thermal slowdown in zero. These readings point to sustained GPU work at the laptop's power limit, with little use of shared GPU memory on this scene. [RTX 4080 results](NVIDIA_GeForce_RTX_4080_presets.csv) · [RTX 4090 results](NVIDIA_GeForce_RTX_4090_presets_current.csv) · [Laptop telemetry by scene](NVIDIA_GeForce_RTX_4050_Laptop_GPU_telemetry.csv)

Across the 36 runs with in-loop telemetry, the largest sampled shared GPU memory use was 113 MiB (Kitchen, Balanced). The largest sampled dedicated GPU memory use was 4.84 GiB (Room, Quality). Windows sampled both counters for the trainer process on the NVIDIA adapter every two seconds. NVML sampled utilization, power, clocks, temperature and PCIe traffic roughly once per second. Each CSV row covers at least 90% of that loop's duration between the first and last samples. Shared GPU memory is system RAM accessible to the GPU. The Windows paging counters in the CSV cover the whole system. [Telemetry CSV](NVIDIA_GeForce_RTX_4050_Laptop_GPU_telemetry.csv) · [Microsoft's GPU memory definitions](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/) · [NVIDIA's clock event definitions](https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlClocksEventReasons.html)

VRAM capacity did limit default Ultrafast on the four indoor Mip-NeRF 360 scenes. At half resolution the trainer's memory planner rejected Bonsai, Counter, Kitchen and Room before step one. After setting aside driver headroom, it had 4.49 GiB available to hold the initial splats, training photos, per-view structure fields and pipeline buffers. [Laptop benchmark coverage](benchmark-rtx4050.md) · [Memory planner](../src/trainer/frontend/app.py)

## Radeon RX 7900 XT

| Preset | Steps | Train loop | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ | Final splats | VRAM at loop end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | 3,000 | 16.44 s | 27.125 dB | 0.7936 | 0.2569 | 1.43M | 5.41 GiB |
| Balanced | 3,000 | 27.02 s | 27.391 dB | 0.8143 | 0.2229 | 2.07M | 2.85 GiB |
| Quality | 3,500 | 39.47 s | 27.898 dB | 0.8286 | 0.1997 | 2.10M | 4.85 GiB |
| Quality+ | 17,189 avg. | 202.04 s | 28.210 dB | 0.8316 | 0.1909 | 2.10M | 4.85 GiB |

Each preset trained on all nine Mip-NeRF 360 scenes at seed 42. Times cover the headless Vulkan training loop through final synchronization. Graphdeco scored 246 held-out RGB8 images per preset on the CPU after training, using the same PSNR, SSIM and VGG LPIPS implementations as the NVIDIA sweeps. The table weights scenes equally. The three fixed-budget PSNR means were within 0.013 dB of their RTX 4090 counterparts. The largest individual scene difference was Room with Ultrafast (−0.136 dB). [AMD per-scene results](AMD_Radeon_RX_7900_XT_presets.csv) · [RTX 4090 per-scene results](NVIDIA_GeForce_RTX_4090_presets_current.csv)

Quality+ reached its training-view plateau on all nine scenes in 8,652–21,996 steps. Compared with Quality, it gained 0.312 dB on average. Its loop time includes convergence checks. All 36 measured runs finished with zero training and held-out sort overflows. [AMD Quality+ per-scene results](AMD_Radeon_RX_7900_XT_qualityplus.csv)

The RX 7900 XT ran on Windows 11 with a 32-core Threadripper PRO 5975WX, display driver 32.0.32029.2010 and SlangPy 0.42.0. The published RTX 4090 ran on a different workstation. On three archived Stump images, CPU scoring reproduced the original CUDA scores to within 0.000002 dB PSNR, 0.0000007 SSIM and 0.0000042 LPIPS. [AMD system and source record](AMD_Radeon_RX_7900_XT_system.txt) · [Reproduction commands](mipnerf360_3k.md#reproduce-the-benchmark)

The fixed-budget sweep included the AMD shader fixes on checkout `676c8e9`. Quality+ ran after those changes were committed at `16679b2`. Both sweeps recorded the same trainer and shader source fingerprint (`b2683e0b7127`).

![Training-loop time, PSNR, SSIM and VGG LPIPS by scene for the three fixed-budget presets on the Radeon RX 7900 XT](assets/rx7900xt-presets.png)

### Differences between independently trained models

Across all four presets, the median absolute per-view PSNR difference between the RX 7900 XT and RTX 4080 was 0.052 dB. Eleven of the 984 comparisons differed by more than 1 dB. The largest difference was Room’s `DSCF4771.JPG` at Quality+. The AMD model scored 29.063 dB and showed a haze over the dark speaker and shelf on the left. The RTX 4080 model scored 33.323 dB. On another Room view, `DSCF4939.JPG`, the AMD model scored 1.612 dB higher. [All matched camera scores and metric differences](AMD_Radeon_RX_7900_XT_vs_RTX_4080_views.csv) · [Room comparison crop](assets/rx7900xt-room-qualityplus-heldout.png)

A second AMD training run with the same seed reached the same 21,996 steps. On that Room view, the haze cleared and PSNR rose to 32.974 dB. Replaying the repeat’s exported PLY on Vulkan reproduced its training render byte-for-byte. Direct3D 12 differed by at most two RGB8 levels at the same camera. This establishes run-to-run variation in the trained result. The original timed run saved images only, so its exact model could not be replayed.

Across the 738 matched fixed-budget views, the median absolute PSNR difference between the RX 7900 XT and RTX 4080 was 0.051 dB. The 95th percentile was 0.365 dB. The largest decrease was 1.774 dB on Kitchen’s `DSCF0712.JPG` at Quality (28.625 dB on AMD, 30.399 dB on NVIDIA). Both reconstructions blurred the background beyond the table. In this view the AMD reconstruction had a visible pale streak along the left table edge. The largest increase was 1.185 dB on Garden’s `DSC08108.JPG` at Balanced.

![Kitchen held-out photograph, RTX 4080 and RX 7900 XT Quality renders, and a map of their squared-error difference. Enlarged crops show the left table edge where the AMD-trained model differed most.](assets/rx7900xt-kitchen-heldout.png)

Two additional seed-42 Quality trainings on the Radeon scored 28.810 dB on that Kitchen view with Vulkan and 29.928 dB with Direct3D 12. Both validation runs are excluded from the benchmark means. At the same camera, replaying one saved Balanced model on the RTX 4080 and RX 7900 XT differed by at most two RGB8 levels. The measurements establish a local difference between trained models. Its cause remains unverified.

### Cross-vendor rendering checks

We rendered two archived RTX 4080 models on the RX 7900 XT using Vulkan. Both cards loaded the same exported PLY and used the same held-out cameras. We checked the model hashes and compared every RGB8 channel at native evaluation resolution. This isolates rendering differences from the differences introduced by training two models independently. [Model hashes and per-scene errors](AMD_Radeon_RX_7900_XT_fixed_model.csv)

| Model | Views | RGB8 channels | Channels differing by more than 1 | Largest difference (out of 255) | PSNR between renders |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stump · Ultrafast | 16 | 49,302,000 | 62 | 5 | 83.86 dB |
| Kitchen · Balanced | 35 | 169,970,010 | 650 | 12 | 82.53 dB |

The mean absolute error was 0.00026 RGB8 levels on Stump and 0.00035 on Kitchen. On a 32 × 32 grid anchored at the top left, the largest regional mean errors were 0.036 and 0.113 levels, respectively. Amplified difference images showed sparse clusters in those regions. To repeat the comparison, run the replay tool on the first card and pass its output as `--reference` when replaying the same model on the second card. [PLY replay comparison tool](../scripts/running/compare_ply_replays.py)

![RTX 4080 and RX 7900 XT crops from Stump and Kitchen. The color scale shows the largest absolute RGB8 channel difference at each pixel. Cyan boxes mark the 32 × 32 regions with the greatest mean error.](assets/rx7900xt-fixed-model-errors.png)

The AMD Vulkan viewer also loaded the 1.59-million-splat Stump PLY. Its 1245 × 825 scene capture at the registered camera matched the headless AMD render in all 3,081,375 RGB8 channels.

Replaying the same two models on AMD Direct3D 12 and AMD Vulkan gave 85.37 dB between their Stump renders and 84.26 dB for Kitchen. Of 219,272,010 RGB8 channels, 278 differed by more than one level, with a maximum difference of 12 levels.

### Training correctness on AMD

Testing exposed a backward-pass bug on 64-lane waves. The rasterizer stores a checkpoint for every 32 splats. Its gradient kernel now treats each half of a 64-lane wave as a separate bucket and reads shuffle values from the correct half. The per-pixel reference kernel sums the number of wave partials that actually ran. Tests compare the different backward kernels and check blend weights against a forward reference on Vulkan and Direct3D 12.

A second fix splits large backward dispatches across two dimensions. Direct3D 12 Balanced training on Kitchen crossed the 65,535-group X limit in all 30 sampled steps. The largest dispatch used 65,535 × 3 groups. The 3,000-step run finished with 1,517,126 splats and zero training or held-out sort overflows. Its 35 saved RGB8 views scored 30.901 dB against the photos, compared with 31.044 dB from an independent Vulkan training run on the same card. Both used seed 42. Direct3D 12 timing is excluded from the Vulkan table.

The 1,202 collected tests passed across fresh test processes on this workstation. A single long-lived pytest process still had a native device-creation failure after more than 1,000 passing cases. The benchmark runs used one fresh training process per scene and preset, as in the published protocol.

## RTX 4090 presets

On the **RTX 4090 24 GB**, Quality averaged **27.90 dB in 22.68 s** with **2.10M final splats**. Compared with the original 3,000-step Quality sweep on the same card, PSNR improved on all nine scenes by an average of 0.113 dB, final splat count fell 31.6%, and loop-end GPU memory fell 21.7%. [Current per-scene measurements](NVIDIA_GeForce_RTX_4090_presets_current.csv) · [Original per-scene measurements](NVIDIA_GeForce_RTX_4090_presets.csv)

| Preset | Steps | Train loop | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ | Final splats | VRAM at loop end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | 3,000 | 10.51 s | 27.13 dB | 0.7935 | 0.2564 | 1.43M | 5.29 GiB |
| Balanced | 3,000 | 15.91 s | 27.40 dB | 0.8141 | 0.2228 | 2.07M | 2.81 GiB |
| **Quality** | **3,500** | **22.68 s** | **27.90 dB** | **0.8285** | **0.1996** | **2.10M** | **4.81 GiB** |
| Quality+ | 17,044 avg. | 122.66 s | 28.227 dB | 0.8317 | 0.1909 | 2.10M | 4.81 GiB |
| Quality (previous) | 3,000 | 23.24 s | 27.79 dB | 0.8274 | 0.2011 | 3.07M | 6.14 GiB |

Quality gained the most PSNR on Bonsai (+0.258 dB) and the least on Stump (+0.012 dB). SSIM improved on seven scenes, with small decreases on Stump (−0.00006) and Treehill (−0.00062). VGG LPIPS improved on seven scenes and rose slightly on Stump (+0.00001) and Treehill (+0.00021). Six scenes trained faster. Counter, Kitchen and Room took 0.31, 0.44 and 0.23 s longer, respectively.

Quality's proposed splits exceeded the growth capacity at 11 of 27 structure updates. All nine runs reached their 75% retention target at step 2,100. All 27 fixed-budget runs completed training and held-out rendering with zero sort overflows. Ultrafast and Balanced used their existing 3,000-step settings. They averaged 11.85 s and 17.31 s in the original sweep and 10.51 s and 15.91 s in this one, with mean PSNR changes of +0.028 and −0.013 dB, respectively.

Quality+ reached its training-view plateau on all nine scenes, taking 8,652–21,996 steps. Compared with 3,500-step Quality, held-out PSNR rose by **0.326 dB** on average, with gains on six scenes. The four indoor scenes gained 0.687 dB on average and the five outdoor scenes gained 0.038 dB. SSIM improved on seven scenes and VGG LPIPS improved on all nine. The loop took 5.41× as long, including convergence checks. [Quality+ per-scene results](NVIDIA_GeForce_RTX_4090_qualityplus.csv) · [Convergence plot](assets/qualityplus-convergence-4090.png) · [Stopping rule](usage.md#quality-convergence)

Each preset trained once per scene at seed 42, using quarter resolution outdoors and half resolution indoors. The clock covers the headless Vulkan training loop and excludes loading, initialization, held-out evaluation and export. VRAM is the trainer process’s local GPU memory sampled at loop end. Graphdeco scored 246 held-out RGB8 renders per preset. Each scene contributes equally to the means after averaging its held-out views.

The sweeps used the same Lenovo workstation with a 32-core Threadripper PRO 9975WX, Windows 11 Enterprise, NVIDIA driver 610.88, SlangPy 0.42.0 and Python 3.11.9. Graphdeco evaluation used PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128. The current trainer revision is `d69f4f2`. [Current system record](NVIDIA_GeForce_RTX_4090_system_current.txt) · [Original system record](NVIDIA_GeForce_RTX_4090_system.txt) · [Reproduce the sweep](benchmark-rtx4090.md)

![Training-loop time, PSNR, SSIM and VGG LPIPS by scene for the three fixed-budget presets on the RTX 4090](../assets/mipnerf360-rtx4090-presets.png)

### Pruning in Ultrafast and Balanced

To test Quality’s post-growth pruning on the two 3,000-step presets, we retrained Bicycle and Kitchen at seed 42 with `--final-splat-fraction 0.75`. Each run used the corresponding default preset for all other training settings. Across those four pairs, pruning reduced final splat count by about 25% and shortened the measured loops by 0.28–0.62 s. SSIM fell and VGG LPIPS rose in every pair. PSNR fell slightly on Bicycle and rose slightly on Kitchen with both presets. Loop-end VRAM differed by at most 4 MiB. [Four pruned runs](mipnerf360_prune75_4090.csv) · [Matched default runs at da6b847](NVIDIA_GeForce_RTX_4090_presets_da6b847.csv)

<a id="current-presets"></a>

## RTX 4080 presets

The updated **Quality** preset averaged **27.90 dB in 32.44 s** on the RTX 4080 16 GB. It improved PSNR on every scene, SSIM on seven and VGG LPIPS on eight. Final model size fell from 3.08M to 2.11M splats on average. The table combines the original Ultrafast and Balanced runs with a new nine-scene Quality sweep. Each row uses seed 42, the stored `images_4` photos outdoors and `images_2` indoors, and all 246 held-out views. Each scene contributes equally to every mean. [Current 4080 CSV](NVIDIA_GeForce_RTX_4080_presets.csv) · [Original 3,000-step CSV](mipnerf360_3k_presets.csv) · [Benchmark commands](mipnerf360_3k.md#reproduce-the-benchmark)

| Preset | Steps | Train loop | Launch to saved PLY | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ | VRAM at loop end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | 3,000 | 17.62 s | 31.28 s | 27.13 dB | 0.7936 | 0.2567 | 5.29 GiB |
| Balanced | 3,000 | 25.97 s | 41.30 s | 27.39 dB | 0.8143 | 0.2229 | 2.80 GiB |
| **Quality** | **3,500** | **32.44 s** | — | **27.90 dB** | **0.8286** | **0.1997** | **4.80 GiB** |
| Quality+ | 17,536 avg. | 216.40 s | — | 28.225 dB | 0.8316 | 0.1907 | 4.80 GiB |
| Quality (previous) | 3,000 | 33.85 s | 49.56 s | 27.78 dB | 0.8275 | 0.2011 | 6.13 GiB |

**Quality+** ran the same Quality schedule for 3,500 updates, then refined its existing splats until training-view PSNR plateaued. All nine scenes reached that stop at seed 42. Against the published 4080 Quality runs, mean held-out PSNR increased by **0.704 dB indoors** and **0.023 dB outdoors**. Six scenes gained PSNR, seven gained SSIM and all nine improved VGG LPIPS. The mean loop took 6.67× as long, including convergence checks. Bicycle and Room informed development. [Per-scene results](NVIDIA_GeForce_RTX_4080_qualityplus.csv) · [Convergence plot](../assets/qualityplus-convergence-4080.png) · [Stopping rule](usage.md#quality-convergence)

**How stable is the stop?** A second Bicycle run at the same seed reached its first plateau at 23,104 updates and 25.844 dB. The published run stopped at 15,668 and 25.800 dB. Continuing the second run with the same splats and optimizer for 12 more camera passes took another 19.23 s and added 0.015 dB on the 25 held-out views. The stopping step can vary appreciably even when the resulting scores are close.

VRAM is the trainer process’s local GPU memory on the RTX 4080, sampled at the end of each headless training loop and recorded in the CSV as `vram_usage_bytes`. Balanced and Quality stream photos from system RAM. Ultrafast keeps the training photos and their structure fields on the GPU for speed.

Quality used one recipe for every scene. It reserved space for 4M splats and 24M splat–tile pairs, prioritized complete splits when requests exceeded capacity (11 of 27 structure updates), and ended growth at step 1,900. At step 2,100 it kept 75% of the live population using the estimated RGB change from removing each splat. The remaining 1,400 steps refined this smaller model. New scenes use the same capacity, fraction and step count. [Settings and implementation](usage.md#training)

The updated loop was faster on eight scenes. Room took 0.68 s longer and gained 0.220 dB. Stump and Treehill had small SSIM decreases (0.00009 and 0.00036), and Bicycle’s VGG LPIPS rose by 0.00006. The scenes informed preset development. The published sweep contains one run per scene and gives no estimate of seed-to-seed variance.

<details>
<summary>Does occlusion help decide which splats to split?</summary>

In the original Quality preset, 27.8% of positive structure votes came from image samples where at most 10% of incoming light reached the splat. Ignoring those samples reduced requested growth by 14.6% when evaluated on the same parents at the same 27 structure updates. In five full retraining runs, the final models were 5.2% smaller on average, but SSIM and LPIPS worsened on all five scenes.

We also tried preserving every proposal and multiplying its gradient-per-child priority by the parent’s mean incoming transmittance over its positive structure votes. We measured transmittance at the four pixels used to interpolate each image-detail sample. Both versions used the same 4M growth capacity, final pruning pass and 3,500 steps.

| Split priority on the RTX 4080 | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ | Train loop | Final splats |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gradient per child (Quality) | 27.8992 | 0.828574 | 0.199660 | 32.44 s | 2.109M |
| Gradient per child × transmittance | 27.8916 | 0.828562 | 0.199404 | 32.32 s | 2.093M |

The transmittance multiplier improved LPIPS on seven scenes and lowered PSNR on seven. Its mean loop-end memory was 61 MiB higher. Quality keeps the simpler split priority and uses occlusion in the later RGB-removal estimate. [Quality measurements](NVIDIA_GeForce_RTX_4080_presets.csv) · [Transmittance experiment](mipnerf360_transmittance_4080.csv)

</details>

![Training-loop time, PSNR, SSIM and VGG LPIPS for the three fixed-budget presets on each of the nine scenes](../assets/mipnerf360-rtx4080-presets.png)

| Preset | Setup + train loop | Launch to model | Launch to closed PLY | Launch to exit | Launch to synced PLY | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | — | 28.30 s | 30.38 s | 31.24 s | 31.28 s | 1.43 M |
| Balanced | — | 36.32 s | 40.39 s | 41.24 s | 41.30 s | 2.07 M |
| Quality | — | — | — | — | — | 2.11 M |
| Quality (previous) | 41.25 s | 43.57 s | — | — | 49.56 s | 3.08 M |

The train loop starts after image loading, stereo initialization and target setup. Launch clocks start at process creation. “Launch to model” ends when the trained Gaussians are ready, before export. “Launch to saved PLY” ends at disk sync. The ultrafast and balanced writers closed the file before process exit, and the timing harness synced it just afterward. The previous Quality sweep synced the PLY before held-out evaluation. Dashes mark clocks without observations on all nine scenes.

The Ultrafast and Balanced scores come from replaying the timed PLY files. The previous Quality sweep saved held-out renders after PLY export. The current Quality sweep trained headless, saved its held-out renders and recorded loop-end memory. Ultrafast uses uncontended timings for the two scenes that initially ran with a separate viewer open. [Saved-model replay check](mipnerf360_3k.md#reproduce-the-benchmark)

The 4080 workstation ran Windows on a 32-core Ryzen Threadripper PRO 5975WX with 128 GiB of RAM and NVIDIA driver 610.60. Python 3.11.9 and SlangPy 0.42.0 ran the Vulkan trainer. Stereo matching ran on the CPU. Both published sweeps used Graphdeco revision `54c035f` and its VGG weights with PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128. All nine updated Quality runs recorded zero sort overflows in training and held-out rendering.

### Where memory goes

On Bicycle, Balanced finished 3,000 steps with 2.49M Gaussians and buffers sized for 2.60M. Quality finished 3,500 steps with 2.60M Gaussians and capacity for 4.00M. Both runs used seed 42 on the headless RTX 4080. SlangPy reports the sizes of named GPU resources. Windows reports the trainer process’s total local GPU memory through DXGI immediately after training, before held-out evaluation or export. [GPU resource accounting](../src/trainer/frontend/evaluation/run_report.py) · [Windows memory counters](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info)

| Bicycle GPU memory at loop end (MiB, rounded) | Balanced | Quality |
| --- | ---: | ---: |
| Rendering, tile sorting, scans and backward-pass workspace | 931 | 1,347 |
| Gradients | 555 | 1,022 |
| Adam optimizer state | 377 | 900 |
| Densification and its workspace | 357 | 687 |
| Gaussian parameters, view-dependent colors and 3D filter | 268 | 580 |
| Loss buffers and current reference photo | 72 | 72 |
| Other process-local GPU usage, including structure workspace | 228 | 229 |
| **Windows process total** | **2,788** | **4,837** |

**Photos.** The nine scenes contain 1,940 original 8-bit RGB JPEGs. Their benchmark-resolution copies are also 8-bit RGB JPEGs. The GPU texture stores four uncompressed bytes per pixel (RGB and alpha). Bicycle has 169 training photos at 1,237 × 822, totaling 656 MiB of packed pixels. Caching the derived structure maps for all 169 views adds another 1,311 MiB before GPU texture padding. Balanced and Quality reuse a single reference texture. Ultrafast caches every training photo and structure map, bringing its Bicycle loop-end usage to 4.16 GiB.

**System RAM.** A separate Balanced Bicycle run on the same RTX 4080 used 3.72 GiB of process working set at loop end and peaked at 4.06 GiB while packing the training photos. The loader retained 2.20 GiB of decoded float32 RGB for all 194 views, including 25 held out, plus 0.64 GiB of packed training photos. Those two allocations account for 2.84 GiB of host image data. [Image decoding](../src/trainer/data/colmap.py) · [Target storage](../src/trainer/data/targets.py)

**What a partial image cache bought on Bicycle.** A prototype kept 63 training photos on the GPU and streamed the remaining 106 through one reusable texture. Both configurations computed structure maps on demand and used the same seed-42 camera order. Caching avoided 1,120 of 3,000 in-loop uploads. In two runs each, in stream/cache/cache/stream order, the mean loop went from 17.00 s to 16.65 s (2.0% shorter), and end VRAM went from 2.72 GiB to 2.99 GiB. The loop excludes preload. The default sampler visits every training camera once per shuffled pass. Replaying the same trace with 64 initially empty least-recently-used slots gave only 231 hits (7.7%). [Camera sampler](../src/trainer/frontend/sampling.py)

**Inspect a run.** With `--run-report`, open `run_report.json` and look under `memory`. The `by_field` map itemizes pipeline GPU allocations. `gpu_view_cache` counts additional per-view GPU textures. `host_ram` gives decoded, packed and linear image-array bytes alongside process working set and peak working set through loop end. The peak spans the lifetime of the process, including previous runs in an open hub. On Windows, `vram_usage_bytes` is the loop-end DXGI sample.

## RTX 4080 comparison

![Training-loop time and held-out PSNR for Gaussian splatting trainers measured on one RTX 4080.](assets/mipnerf360-rtx4080-frontier.png)

The dashed line connects configurations that improve on the best PSNR reached at a shorter training time. Each horizontal tick doubles the training-loop budget. Quality+ extends the curve to 28.225 dB at 216.40 s on the same card. [Quality+ per-scene measurements](NVIDIA_GeForce_RTX_4080_qualityplus.csv)

| External trainer | Recipe | Mean train loop | Mean PSNR |
| --- | --- | ---: | ---: |
| LiteGS | Aggressive, 1M target splats, 10,000 requested steps | 59.27 s | 27.58 dB |
| SAD-GS | 3,000 steps, float32 structure maps cached in system RAM | 159.21 s | 27.19 dB |
| SkipGS on FastGS Standard | 30,000 steps, backward skipping from step 15,000 | 184.26 s | 27.43 dB |
| FastGS-Big | 30,000 steps, authors’ per-scene Big settings | 325.86 s | 27.83 dB |

In these runs, PERF-GS Quality scored **27.8992 dB in 32.44 s** and FastGS-Big scored **27.8345 dB in 325.86 s**. LiteGS completes whole passes through the training views, yielding 9,760–9,982 updates across scenes. SAD-GS uses two views per indoor iteration and one outdoors. Transfers from its host cache count toward the loop. On bicycle, a densification step spilled into shared GPU memory on this 16 GB card.

SkipGS uses FastGS Standard’s densification settings. It skipped 50,426 of 135,009 backward passes from its activation point across the nine scenes. FastGS-Big uses a different densification recipe.

All trainers ran headless on the same RTX 4080 16 GB, with one seed-42 run per scene. PERF-GS used Windows and Vulkan. The other trainers used WSL2 and CUDA. The wall clock starts after loading and initialization, includes the final GPU synchronization, and stops before model export or evaluation. [Per-scene PERF-GS data](NVIDIA_GeForce_RTX_4080_presets.csv) · [LiteGS data](mipnerf360_litegs_4080.csv) · [Additional trainers](mipnerf360_peers_4080.csv)

All runs used the stored `images_4` outdoors and `images_2` indoors. Each trainer’s native renderer produced the 246 held-out views. Graphdeco’s RGB8 PSNR uses identical reference pixels for each method. Views contribute equally within each scene, and the nine scene means contribute equally to each point. [Reproduce LiteGS](mipnerf360_3k.md#compare-litegs-on-the-same-gpu) · [Reproduce the other trainers](mipnerf360_3k.md#reproduce-the-additional-trainers)

## Published Mip-NeRF 360 comparisons

These are the authors’ reported Mip-NeRF 360 means. SAD-GS used an H100, with a batch of two training images per indoor step and one per outdoor step. Its released evaluator at `029814f` uses the same RGB8 PSNR, SSIM and VGG LPIPS calculations as Graphdeco. Its published training script selects `images` with automatic sizing, so the exact source-image dimensions for the paper run remain unverified. PERF-GS uses one training image per step. The original 3DGS paper reports two larger training budgets on an RTX A6000. Hardware, software and timing boundaries differ across papers. [SAD-GS, Table 1 and Section 4.1](https://arxiv.org/html/2604.28016v1#S4.T1) · [3DGS, Table 1](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf)

| Published method | Steps | GPU | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time reported |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| SAD-GS | 3,000 | H100 | 27.25 dB | 0.821 | 0.197 | 52.96 s optimization |
| Original 3DGS | 7,000 | RTX A6000 | 25.60 dB | 0.770 | 0.279 | 6 min 25 s training |
| FastGS | 30,000 | RTX 4090 | 27.56 dB | 0.797 | 0.261 | 1.93 min training |
| Original 3DGS | 30,000 | RTX A6000 | 27.21 dB | 0.815 | 0.214 | 41 min 33 s training |
| EDGS + 3DGS | 30,000 | A100 | 28.02 dB | 0.839 | 0.141 | 27 min including initialization |
| dBS (MCMC) | 30,000 | A100 | 28.70 dB | 0.842 | 0.185 | 24.42 min training |

EDGS and dBS report averages over all nine scenes. The EDGS benchmark uses quarter-resolution outdoor images and half-resolution indoor images. Its evaluator writes two VGG LPIPS scores, and the paper reports one without identifying which. dBS specifies joint-RGB PSNR and leaves the LPIPS backbone unspecified. [EDGS, Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Kotovenko_EDGS_Eliminating_Densification_for_Efficient_Convergence_of_3DGS_CVPR_2026_paper.pdf#page=5) · [EDGS benchmark](https://github.com/CompVis/EDGS/blob/f90b022445fc88368f75e66e8fb34aea88372cac/full_eval.py) · [EDGS evaluator](https://github.com/CompVis/EDGS/blob/f90b022445fc88368f75e66e8fb34aea88372cac/metrics.py) · [dBS results and protocol](https://arxiv.org/html/2610.22778v1#S5.SS1)

Opt3DGS reports 29.96 dB on seven scenes, omitting flowers and treehill. Averaging the updated 3,500-step RTX 4080 Quality runs over those seven gives 29.40 dB, versus 27.90 dB over all nine. [Opt3DGS, §5 and Table 1](https://ojs.aaai.org/index.php/AAAI/article/download/37438/41400#page=5) · [PERF-GS per-scene scores](NVIDIA_GeForce_RTX_4080_presets.csv)

SAD-GS reports 4.05 million final Gaussians at 3,000 steps. FastGS reports 0.40 million at 30,000 steps. [SAD-GS, Table 1](https://arxiv.org/html/2604.28016v1#S4.T1) · [FastGS, Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Ren_FastGS_Training_3D_Gaussian_Splatting_in_100_Seconds_CVPR_2026_paper.pdf)
