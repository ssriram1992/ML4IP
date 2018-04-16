#!/bin/bash -l
#SBATCH --job-name=Feat100
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=23:0:0
#SBATCH --mail-type=end
#SBATCH --mail-user=srirams@jhu.edu

module unload git
module load anaconda-python/3.5.2
python -c 'import cplex'
# Use this place to call the program you want to run!
python genFeat.py List1
module unload anaconda-python/3.5.2

