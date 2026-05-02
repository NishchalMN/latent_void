# Decision Log

Last updated: 2026-05-02

## D001: Treat `latent_void` As A Standalone Repo

Decision:
`/fs/nexus-scratch/gnanesh/DiffSplat/latent_void` is the project root and Git
repo. Do not treat it as a subfolder of a parent DiffSplat repo.

Reason:
The parent `.git` was removed to avoid accidental commits of unrelated files.
This keeps code, configs, docs, Slurm scripts, and tests scoped to this project.

## D002: Use Real Scene Data First

Decision:
Start with Inpaint360GS scenes rather than DiffSplat text-to-3D generated
objects.

Reason:
DiffSplat generation is object-centric and not a good source for full
indoor/outdoor environments. The intended task is scene object removal and
completion.

## D003: Use DiffSplat As Scene Encoder

Decision:
Use DiffSplat `GSRecon` and GSVAE-style representation as the scene encoding
path.

Reason:
This preserves the desired latent-space pipeline while allowing real multi-view
scene data to be converted into structured Gaussian grids.

## D004: Use SAM 3 For Multi-View Prompt Masks

Decision:
SAM 3 provides object and optional shadow/effect masks across calibrated scene
views.

Reason:
Prompted multi-view segmentation is necessary to create a true 3D deletion
mask before latent inpainting.

## D005: Avoid Per-View 2D Inpainting As Main Method

Decision:
The main synthesis path is native latent/Gaussian inpainting.

Reason:
Independent 2D inpainting creates cross-view inconsistencies, geometric drift,
and projection artifacts. It can be used for diagnostics or ablations later, but
not as the core pipeline.

## D006: Keep Heavy Models Behind Command Adapters

Decision:
Do not vendor DiffSplat, SAM 3, datasets, or checkpoints in this repo.

Reason:
The local workspace should remain lightweight. Zaratan H100 jobs will call
configured external repositories and checkpoints.

## D007: Use `msml612pcs3-class` By Default On Zaratan

Decision:
Slurm templates default to `msml612pcs3-class` and `gpu-h100`.

Reason:
The Zaratan balance showed substantially more unused allocation there than the
other visible account.

## D008: Parse Inpaint360GS COLMAP Metadata Locally

Decision:
The dataset loader reads COLMAP `sparse/0` text or binary cameras/images files
and stores camera intrinsics plus camera-to-world poses in the view manifest.

Reason:
Inpaint360GS scenes are real multi-view sequences with COLMAP calibration, not
prepacked DiffSplat manifests. Keeping this parser local makes dataset discovery
and later GSRecon adapters reproducible.

## D009: Treat GSRecon Scene Export As A Real Adapter Task

Decision:
`tools/run_gsrecon_export.py` stays a failing contract wrapper until we build a
proper DiffSplat adapter for Inpaint360GS.

Reason:
The public DiffSplat repo does not ship a direct scene export CLI, and the
pretrained GSRecon checkpoint expects GObjaverse-style RGB plus geometry
channels. Returning fake Gaussian grids would hide the main research risk.
