#!/bin/sh
 
#SBATCH --job-name=train_RGM
#SBATCH --output=train_RGM-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=train_RGM-%A.err  # Standard error of the script
#SBATCH --time=0-48:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gpus=1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)
#SBATCH --nodelist=ithor
 
# run the program
ml cuda  # load default CUDA module
ml miniconda3  # load default miniconda and python module
source activate MA-RGM # enter conda env
python train.py --cfg experiments/train_RGM_Crop_ribseg_2048_transformer.yaml
ml -cuda -miniconda3  # unload all modules