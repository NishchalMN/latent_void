# Latent Void: 3D Generative Scene Editing & Object Removal

**Latent Void** is an end-to-end 3D scene editing pipeline that seamlessly removes complex objects from real-world environments by bridging **3D Gaussian Splatting (3DGS)** radiance fields with **2D Generative AI** foundation models.

By integrating zero-shot segmentation (**SAM 3**) and latent inpainting (**LaMa**) with custom PyTorch loss functions, Latent Void natively eliminates geometry collapse, rendering artifacts, and monocular depth misalignments, achieving photorealistic, artifact-free 360-degree novel camera trajectories.

## 🚀 Results

Below are the 3D orbit renderings of a real-world scene (a bag on a street), demonstrating the progression from the original 3DGS scene to the masked void, and finally the generative inpainting result.

### 1. Original 3DGS Scene
<video src="with_bag.mp4" autoplay loop muted playsinline width="800"></video>

### 2. Masked / Pruned 3D Void (Target Object Removed)
<video src="masked.mp4" autoplay loop muted playsinline width="800"></video>

### 3. Generative 3DGS Filled Scene
<video src="filled.mp4" autoplay loop muted playsinline width="800"></video>

---

## 🛠 Architecture & Highlights

- **End-to-End Pipeline:** Reconstructs scenes via `GSRecon`, dynamically prunes targets via SAM 3, and utilizes LaMa for high-fidelity latent inpainting.
- **Geometry-Aware Fusion:** Projects 2D inpainted textures back into 3D space. Corrects monocular depth misalignment with affine transformations to guarantee strict multi-view depth consistency.
- **Optimized Rendering Losses:** Re-architected `masked_l1_loss`, SSIM, and LPIPS to strictly supervise missing regions without disrupting surrounding geometry, completely eliminating streaking and floating artifacts during finetuning.

## ⚙️ Quick Start

This repository intentionally keeps heavyweight model code behind adapters so local development can validate configs, masks, latent shapes, and command orchestration without needing to install everything.

```bash
# Validate configs
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m unittest discover -s tests
```

On HPC / Zaratan:
```bash
git pull
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
scripts/zaratan_srun_stage.sh geometry configs/zaratan_inpaint360gs_bag.yaml \
  --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_srun_h100
```
