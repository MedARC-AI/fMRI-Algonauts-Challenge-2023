#!/bin/bash
# args=("${@:2}")
args=("$@")
args=("${args[@]:2}")

sbatch << EOT
#!/bin/bash
#SBATCH --account topfmri
#SBATCH --cpus-per-gpu 8                # Number of cores
#SBATCH -N 1                    # Force single node
#SBATCH --time 20:00:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --partition g40         # Partition to submit to
#SBATCH --gpus 1        # Number of GPUs
#SBATCH --mem 64000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../train_logs/slurm/$2.out  # %j inserts jobid
#SBATCH -e ../train_logs/slurm/$2.err  # %j inserts jobid
#SBATCH --exclude=ip-26-0-129-240,ip-26-0-129-157,ip-26-0-131-188

export NUM_GPUS=1
echo NUM_GPUS=\${NUM_GPUS}

export MASTER_PORT=\$((RANDOM % (19000 - 11000 + 1) + 11000))
export HOSTNAMES=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST")
echo HOSTNAMES=\${HOSTNAMES}
export MASTER_ADDR=\$(echo "\$HOSTNAMES" | head -n 1)
export COUNT_NODE=\$(echo "\$HOSTNAMES" | wc -l)
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM=false

export WANDB_DIR="/fsx/proj-medarc/fmri/fMRI-reconstruction-NSD/src/wandb/"
export WANDB_CACHE_DIR="/fsx/proj-medarc/fmri/atom/.cache"
export WANDB_MODE="online"

echo MASTER_ADDR=\${MASTER_ADDR}
echo MASTER_PORT=\${MASTER_PORT}
echo WORLD_SIZE=\${COUNT_NODE}
echo all_args=${@}
echo args=${args[@]}

# Ignore SIGTERM signal
trap '' SIGINT SIGTERM SIGCONT

module load cuda/11.7
/admin/home-atom_101/python_envs/fmri/bin/accelerate launch --num_machines \${COUNT_NODE} --num_processes \$((NUM_GPUS * COUNT_NODE)) --main_process_ip \${MASTER_ADDR} --main_process_port \${MASTER_PORT} --gpu_ids='all' --dynamo_backend='no' $1 --model_name=$2 ${args[@]}
EOT
