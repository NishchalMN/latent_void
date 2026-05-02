# Context

Last updated: 2026-05-02

## Goal

Build `latent_void` as a standalone repo for native 3D object removal from real
multi-view scenes. The system should remove a prompted object using SAM 3 masks
from multiple views, create a true 3D void, project that void into
DiffSplat/GSVAE latent space, inpaint it natively, decode back to Gaussian
scene representation, and render a cleaned scene without obvious artifacts.

## Core Understanding

The project should not use DiffSplat's text-to-object generation as the base
scene source. DiffSplat is object-centric and trained around G-Objaverse-style
assets, so it is not the right source for full indoor/outdoor environments.

Instead, real multi-view scene data should be reconstructed into DiffSplat's
structured representation:

1. Load Inpaint360GS scenes first.
2. Pass calibrated multi-view images through DiffSplat `GSRecon`.
3. Produce GSVAE-compatible 12-channel Gaussian grids.
4. Encode grids into 4-channel GSVAE splat latents.
5. Use SAM 3 across views to segment objects and optional shadows.
6. Fuse masks into Gaussian-level deletion masks.
7. Convert deletion masks into latent void masks.
8. Inpaint directly in latent/Gaussian space.

## Research Direction

The preferred method is native 3D/latent inpainting, not independent per-view 2D
inpainting followed by reprojection. Per-view 2D inpainting is allowed only as a
diagnostic or fallback comparison, not as the main synthesis method.

The later training path is self-supervised masked fine-tuning:

- Use large intact real-world datasets such as DL3DV-10K.
- Randomly mask 3D regions or object-like volumes.
- Train the model to reconstruct the original intact scene.
- Avoid requiring paired "object present" and "object removed" datasets.

## Current Scope

The current code is an MVP scaffold and orchestration layer. It does not vendor
DiffSplat, SAM 3, Inpaint360GS, or pretrained checkpoints. Heavy model work is
plugged in through config-driven command adapters and run on Zaratan H100s.

The local repo should be able to:

- Validate configs.
- Discover scene files.
- Render dry-run external commands.
- Fuse masks into deletion masks when projected Gaussian visibility data exists.
- Generate latent void masks.
- Run a plumbing fallback latent fill for tests.

The real model path depends on configured DiffSplat, SAM 3, dataset, and
checkpoint paths on Zaratan.
