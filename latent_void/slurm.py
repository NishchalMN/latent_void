from latent_void.config import get_nested


def sbatch_header(config, job_name):
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=%s" % job_name,
        "#SBATCH --account=%s" % get_nested(config, "hpc.account", "msml612pcs3-class"),
        "#SBATCH --partition=%s" % get_nested(config, "hpc.partition", "gpu-h100"),
        "#SBATCH --gres=%s" % get_nested(config, "hpc.gres", "gpu:h100:1"),
        "#SBATCH --time=%s" % get_nested(config, "hpc.time", "02:00:00"),
        "#SBATCH --cpus-per-task=%s" % get_nested(config, "hpc.cpus_per_task", 8),
        "#SBATCH --mem=%s" % get_nested(config, "hpc.mem", "64G"),
        "#SBATCH --output=logs/%x-%j.out",
        "#SBATCH --error=logs/%x-%j.err",
    ]
    return "\n".join(lines) + "\n"
