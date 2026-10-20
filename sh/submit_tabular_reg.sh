#!/bin/bash
#SBATCH --job-name=Tabular_Bemchmarking_regression
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=                # Where to send mail
#SBATCH --account=            # Define Account
#SBATCH --qos=                  # Define qos
#SBATCH --cpus-per-task=10            # Define the cpu for task	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=200gb  		      # Memory
#SBATCH --gpus=1                       # GPU 
#SBATCH --time=10-00:00:00              # Time limit hrs:min:sec
#SBATCH --output=/regression%j.log
pwd; hostname; date

module load conda
conda activate tabular311
cd /Experiments
export HF_TOKEN=""
python regression_benchmark.py
date
