# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Load a .ply file and display it in the interactive viewer.

    .venv/Scripts/python.exe scripts/debug/view_ply.py path/to/output.ply --width 1280 --height 720
"""

from __future__ import annotations

import argparse

from trainer.data.ply import load_ply
from trainer.common.sh import sh_degree_from_rest_floats
from trainer.config.config import AppearanceConfig
from trainer.backend.compute.device import create_device
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.frontend.viewer.viewer import Viewer

def main() -> None:
    ap = argparse.ArgumentParser(description="View a .ply gaussian splat file")
    ap.add_argument("ply", help="path to the .ply file to view")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max-splats", type=int, default=1_000_000)
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    args = ap.parse_args()

    data = load_ply(args.ply)
    n = data["means"].shape[0]
    print(f"loaded {n} gaussians from {args.ply}")

    rest = data.get("features_rest")
    sh_degree = sh_degree_from_rest_floats(rest.shape[1]) if rest is not None else 0
    if sh_degree > 0:
        print(f"  SH degree {sh_degree} ({rest.shape[1]} f_rest coeffs) -> view-dependent")

    device = create_device(backend=args.backend)
    pipeline = TrainingPipeline(device=device, width=args.width, height=args.height,
                        max_splats=max(args.max_splats, n),
                        appearance_config=AppearanceConfig(sh_degree=sh_degree),
                        initial_state=data, viewer_only=True)
    if sh_degree > 0:
        pipeline.set_active_sh_degree(data.get("active_sh_degree", sh_degree))

    viewer = Viewer(device, width=args.width, height=args.height,
                    max_splats=max(args.max_splats, n), hot_reload=True,
                    sh_degree=sh_degree, mip_enabled=pipeline.raster_config.mip_enabled,
                    mip_kernel_size=pipeline.raster_config.kernel_size)
    viewer.update_gaussians(pipeline)
    # Static view: suppress the panel's "press Start Training" hint.
    viewer.set_training_active(True)

    print("viewer running — close the window to exit")
    viewer.run()


if __name__ == "__main__":
    main()
