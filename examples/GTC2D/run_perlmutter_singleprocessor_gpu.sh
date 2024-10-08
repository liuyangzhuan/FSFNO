#!/bin/bash -l

#SBATCH --account=m4484
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:59:00
#SBATCH --job-name=fno-train
#SBATCH --mail-user=rahman@lbl.gov 
#SBATCH --mail-type=ALL
#SBATCH -C 'gpu&hbm80g'


# module load cpe/23.03 pytorch/1.13.1
# module load cpe/23.03
# module load pytorch/1.13.1
# export LD_LIBRARY_PATH=/global/common/software/nimrod/perlmutter/multispecies-gcc-11.2/nimdevel-jking-main-v1.0-dev.r287/lib/:$LD_LIBRARY_PATH ;
module load python ; module load pytorch 

export PYTHONPATH=$PWD/../../stfno/:$PYTHONPATH

python main.py