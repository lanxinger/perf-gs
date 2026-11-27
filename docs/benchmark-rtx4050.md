# Reproduce the RTX 4050 Laptop GPU benchmark

The three fixed-budget presets cover 13 scenes from the 3DGS benchmark datasets. Each scene uses one headless Vulkan process at seed 42. Every eighth camera in filename order is held out. The clock covers the optimization loop through final GPU synchronization. Loading, initialization, held-out rendering, scoring and export are outside that clock.

| Dataset | Scenes | Image scale | Test views per preset |
| --- | --- | --- | ---: |
| Mip-NeRF 360 | Bicycle, Flowers, Garden, Stump, Treehill, Bonsai, Counter, Kitchen, Room | Stored quarter resolution outdoors and half resolution indoors | 246 |
| Tanks & Temples | Train, Truck | Half resolution | 70 |
| Deep Blending | DrJohnson, Playroom | Full resolution | 62 |

Graphdeco scores RGB8 PSNR, SSIM and VGG LPIPS for each held-out image. Average the images within each scene, then average the scenes within each dataset with equal weights. VRAM in the results table is the trainer process's local GPU memory at loop end. [Per-scene measurements and settings](NVIDIA_GeForce_RTX_4050_Laptop_GPU_benchmarks.csv) · [Truck timing and memory investigation](benchmarks.md#truck-timing-and-memory)

## Setup

Use the repository's Windows benchmark environment with SlangPy, PyTorch, torchvision, the official sampling masks and both Mip-NeRF 360 archives. Install the Graphdeco evaluator at the same revision used for the desktop sweeps. Run these commands from the repository root with the virtual environment activated. [Environment setup and mask download](benchmark-rtx4090.md#set-up-once)

```powershell
python scripts/data/download_tandt_db.py --set all
$reference = 'outputs/graphdeco_reference'
if (!(Test-Path $reference)) {
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git $reference
}
git -C $reference checkout --detach 54c035f7834b564019656c3e3fcc3646292f727d
```

## Train and score the published configurations

Close the viewer and other GPU workloads. The CSV records the requested preset, any image-residency override and the splat capacity the trainer could fit. The commands below use a fresh output directory for each sweep. To test the stock preset on a different GPU, omit the image-residency override. Local memory budgets and laptop power limits can change the capacity and time.

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$models = @()
foreach ($row in (Import-Csv docs/NVIDIA_GeForce_RTX_4050_Laptop_GPU_benchmarks.csv)) {
    $scene = $row.scene
    $group, $scale = switch -Regex ($scene) {
        '^(bicycle|garden|stump)$' { '360_v2', 4 }
        '^(flowers|treehill)$'     { '360_extra_scenes', 4 }
        '^(bonsai|counter|kitchen|room)$' { '360_v2', 2 }
        '^(train|truck)$'          { 'tandt', 2 }
        '^(drjohnson|playroom)$'   { 'db', 1 }
    }
    $name = "repro_${stamp}_$($row.preset)_$scene"
    $run = "outputs/train/$name"
    $model = "outputs/graphdeco_input/$name"
    $flags = @('--no-write-ground-truths')
    if ($group -in 'tandt','db') { $flags = @('--write-ground-truths') }
    if ($row.image_residency_override) { $flags += '--image-residency', $row.image_residency_override }
    python -m trainer.frontend.app --scene "datasets/$group/$scene" --downsample $scale `
        --preset $row.preset --iters $row.steps --seed 42 --backend vulkan --no-viewer `
        --evaluation --final-test-psnr --preview-every 0 --eval-every 0 --log-every 0 `
        --run-report --save-test-renders --run-name $name @flags
    if ($LASTEXITCODE) { throw "Training failed for $name" }
    if ($group -in '360_v2','360_extra_scenes') {
        python scripts/running/prepare_graphdeco.py $run $model
        if ($LASTEXITCODE) { throw "Preparation failed for $name" }
    } else {
        $report = Get-Content -Raw "$run/run_report.json" | ConvertFrom-Json
        $method = "$model/test/ours_$($report.iters_completed)"
        New-Item -ItemType Directory -Force "$method/renders","$method/gt" | Out-Null
        $ordinal = 0
        foreach ($view in $report.quality.per_view) {
            $filename = '{0:D5}.png' -f $ordinal
            $cam = 'cam_{0:D2}' -f $view.cam_idx
            Copy-Item "$run/test_renders/$($view.render_png)" "$method/renders/$filename"
            Copy-Item "$run/cams/$cam/gt.png" "$method/gt/$filename"
            $ordinal++
        }
    }
    $models += $model
}
python scripts/running/score_graphdeco.py --graphdeco $reference @models
```

For Mip-NeRF 360, the evaluator pairs saved renders with the stored benchmark-resolution photographs. For Tanks & Temples and Deep Blending it pairs renders with the trainer's RGB8 targets at the recorded camera size. The reproduction command writes those targets before training. The published sweep regenerated them with the same COLMAP image loader and RGB8 conversion. Graphdeco writes scene means to each model's `results.json` and individual image scores to `per_view.json`.
