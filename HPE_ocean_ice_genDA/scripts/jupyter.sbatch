#!/bin/bash
#SBATCH --job-name=jlab               # Job name
#SBATCH --nodes=1
#SBATCH --time=04:00:00               # Time limit hrs:min:sec

port=45119 
cd /home/users/e45119/GenDA/HPE_ocean_ice_genDA || exit 1

ssh -N -f -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:${port}:127.0.0.1:${port}" portage-login1 || exit 1

pixi run -e gpu-dev-protage jupyter lab \
    --no-browser --ip=127.0.0.1 --port="$port" --port-retries=0