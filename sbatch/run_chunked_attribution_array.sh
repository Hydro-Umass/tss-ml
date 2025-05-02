#!/bin/bash
#SBATCH --job-name=chunked_integrated_gradients
#SBATCH --output=/work/pi_kandread_umass_edu/tss-ml/runs/CONUS/_slurm_outputs/attribution_chunks_%A_%a.out
#SBATCH --array=0-63
#SBATCH -t 1-00:00:00
#SBATCH -p gpupod-l40s
#SBATCH -q gpu-quota-16
#SBATCH -c 2
#SBATCH --gpus=1 # Request access to 1 GPU
#SBATCH -A pi_cjgleason_umass_edu
#SBATCH --mem=32G

source .venv/bin/activate

module load cuda/12.6
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

SCRIPT_PATH="/work/pi_kandread_umass_edu/tss-ml/src/run.py"
MODEL_PATH="/work/pi_kandread_umass_edu/tss-ml/runs/CONUS/train_all_sites_20250428_200346"

# Run the Python script with the current array task ID as the grid search index
python $SCRIPT_PATH --chunked_attribution $MODEL_PATH --chunk_index $SLURM_ARRAY_TASK_ID --n_chunks 64
