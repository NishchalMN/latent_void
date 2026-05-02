# Zaratan Runbook

## One-Time Repo Setup

The `zaratan` tmux session already has the working copy at:

```bash
/home/gnanesh/scratch.msml612pcs3/latent_void
```

If the clone is empty or stale:

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
git remote -v
git pull --ff-only
```

If SSH authentication is unavailable on Zaratan:

```bash
git remote set-url origin https://github.com/NishchalMN/latent_void.git
git pull --ff-only
```

## Smoke Job

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
sbatch slurm/zaratan_smoke.sbatch configs/inpaint360gs_example.yaml
```

The smoke job only validates config, dataset discovery, and dry-run command
rendering. It should not spend meaningful GPU time.

## Real Job

Copy and edit the example config with real checkpoint and dataset paths:

```bash
cp configs/inpaint360gs_example.yaml configs/my_scene.yaml
```

Then run:

```bash
sbatch slurm/zaratan_inpaint.sbatch configs/my_scene.yaml
```

Default Slurm settings:

- account: `msml612pcs3-class`
- partition: `gpu-h100`
- GRES: `gpu:h100:1`
