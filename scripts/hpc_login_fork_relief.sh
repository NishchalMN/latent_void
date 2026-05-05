#!/bin/sh
# When the login node prints: bash: fork: Resource temporarily unavailable
# the kernel has refused to create new processes (often nproc / memory).
#
# Do this in order:
# 1) Open a NEW SSH session from your laptop (do not nest tmux/screen deeply).
# 2) Close extra Cursor/IDE terminals and long-running watch loops on the login node.
# 3) Run this script with: /bin/sh scripts/hpc_login_fork_relief.sh
# 4) If still stuck, wait 5–15 minutes or ask cluster support; avoid spawning
#    parallel job arrays or Python multiprocessing on the login node.
#
# Heavy training belongs in srun/tmux on a GPU node, not on the login node.

printf '%s\n' "--- fork pressure snapshot ---"
ulimit -u 2>/dev/null && printf '%s\n' "(soft max user processes)"
ps -u "${USER:-$(id -un)}" -o pid= 2>/dev/null | wc -l | awk '{print "Your processes:", $1}'
printf '%s\n' "Tip: export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 on login-node checks."
