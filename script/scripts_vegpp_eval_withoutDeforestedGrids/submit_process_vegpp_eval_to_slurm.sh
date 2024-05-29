#!/bin/bash

#
########## Begin Slurm header for resources ##########
#
#SBATCH --job-name=postprocessing_sindbad
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=$USER@bgc-jena.mpg.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=postprocessing_sindbad.%j.o.log
#SBATCH --error=postprocessing_sindbad.%j.e.log
#SBATCH --mem-per-cpu=256G
########### End Slurm header for resources ##########

# ulimit -s unlimited
path_to_bashfile="/Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withoutDeforestedGrids/process_vegpp_eval.sh"
bash $path_to_bashfile 

exit 0