# Zaratan Memory

Last updated: 2026-05-02

## Known Paths

Local repo:

```bash
/fs/nexus-scratch/gnanesh/DiffSplat/latent_void
```

Zaratan repo:

```bash
/home/gnanesh/scratch.msml612pcs3/latent_void
```

GitHub:

```bash
https://github.com/NishchalMN/latent_void.git
git@github-nishchal:NishchalMN/latent_void.git
```

## Known Cluster State

Visible accounts from `sbalance`:

- `msml605-class`
- `msml612pcs3-class`

Use `msml612pcs3-class` by default.

Useful partitions from `sinfo`:

- `gpu-h100`
- `gpu`
- `gpu-a100`
- `gpu-v100`

Use `gpu-h100` for heavy jobs.

## Sync Workflow

Local:

```bash
git status --short --branch
git add .
git commit -m "..."
git push
```

Zaratan tmux session:

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
git pull --ff-only
git status --short --branch
```

Helper from local machine:

```bash
scripts/pull_zaratan.sh
```

## SSH Note

GitHub SSH clone on Zaratan failed with public-key authentication. HTTPS clone
and pull worked. If the remote ever points at SSH on Zaratan, switch it back:

```bash
git remote set-url origin https://github.com/NishchalMN/latent_void.git
```

## Login-Node Python Note

Zaratan login-node `python3` had:

- `yaml`: available
- `numpy`: unavailable

The CLI dry-run path was adjusted so config validation and dry-run orchestration
do not import NumPy. Real `fuse` and `inpaint` stages still need NumPy in the
active GPU/model environment.

## Smoke Commands

```bash
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m latent_void run --config configs/inpaint360gs_example.yaml --dry-run
sbatch slurm/zaratan_smoke.sbatch configs/inpaint360gs_example.yaml
```

## Real Job Command

After creating a real config:

```bash
sbatch slurm/zaratan_inpaint.sbatch configs/zaratan_inpaint360gs_scene.yaml
```
