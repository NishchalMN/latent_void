## Latent Void: Project Overview

### Objective
Latent Void is a research pipeline for native 3D object removal in real multi-view scenes. The goal is to remove an object from a reconstructed 3D scene and inpaint the missing region directly in DiffSplat/GSVAE latent space, rather than using per-view 2D inpainting followed by reprojection.

The first target dataset is Inpaint360GS, with future scaling planned through larger calibrated scene datasets such as DL3DV.

### Core Idea
The project adapts DiffSplat, which was originally trained mostly on object-centric GObjaverse data, to real scene-local patches. The intended flow is:

1. Load a calibrated multi-view scene.
2. Generate geometry side channels: RGB, depth, normals, coordinate maps, cameras.
3. Run DiffSplat GSRecon/GSVAE to produce Gaussian grids and latent scene representations.
4. Segment the target object using SAM 3 across multiple views.
5. Fuse masks into a 3D/latent void mask.
6. Inpaint the missing region in latent space.
7. Decode back to Gaussians, merge into the scene, and render before/after views.

### Why Scene-Local Adaptation Is Needed
A key finding is that pretrained DiffSplat works on official GObjaverse object examples, but performs poorly on full real-world Inpaint360GS scenes. The problem is domain mismatch: DiffSplat expects object-centric normalized inputs, while real scenes are larger, messier, and more geometrically complex.

To address this, the project now uses a scene-local adaptation layer: instead of feeding entire scenes directly into DiffSplat, it extracts local object-scale patches with RGB, masks, depth, normals, coordinate maps, cameras, and held-out views.

### What Has Been Implemented
The current codebase includes:

- Dataset discovery for Inpaint360GS and a DL3DV-style loader scaffold.
- Geometry preprocessing using Marigold depth/normals and COLMAP cameras.
- DiffSplat GSRecon/GSVAE export adapters.
- SAM 3 multi-view segmentation.
- Mask fusion into Gaussian and latent void masks.
- Local patch extraction and canonicalization.
- Teacher target generation from held-out patch views.
- Masked latent denoiser training.
- Reconstruction adapter training.
- Visual diagnostics and evaluation reports.
- Local Gaussian merge/compositing scaffolding.

### Current Training Status
Two scaffold training paths have been run on available Inpaint360GS data.

The masked latent denoiser was trained on synthetic masks over local DiffSplat latents. It successfully learned to fill masked latent cells while preserving unmasked cells exactly.

The reconstruction adapter was trained on local patches from 11 Inpaint360GS scenes, using held-out RGB/alpha/depth views as teacher targets. This confirmed that multi-scene patch data generation and reconstruction supervision work, though this is still a lightweight adapter rather than full DiffSplat fine-tuning.

### Current Limitation
The project is not yet producing final-quality inpainted 3D scene renders. The main blocker is still real-scene reconstruction quality. Before inpainting can be judged visually, DiffSplat/GSRecon needs to produce readable local scene patch renders on real data.

### Next Steps
The next research step is to move from scaffold adapter training to actual DiffSplat GSRecon/GSVAE adaptation on scene-local patches. Once reconstruction quality improves, the trained latent denoiser can be applied to real object voids, decoded back into Gaussians, merged into the full scene, and rendered as before/void/inpainted comparisons.

### Summary
Latent Void has progressed from pipeline scaffolding to real GPU-backed training and evaluation on multi-scene Inpaint360GS patches. The latent inpainting mechanics are working, and the scene-local data pipeline is in place. The remaining core challenge is adapting DiffSplat’s reconstruction model from object-centric data to complex real scenes.