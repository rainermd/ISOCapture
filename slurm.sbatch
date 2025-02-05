#!/bin/bash
#SBATCH - PACETest                              # Job name
#SBATCH --account=hive-gli323                   # Tracking account
#SBATCH -n4                                     # Number of cores required
#SBATCH --mem-per-cpu=1G                        # Memory per core
#SBATCH -t15                                    # Duration of the job (Ex: 15 mins)
#SBATCH -phive                                  # Queue name (where job is submitted)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=rmarquar3@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3                           # Load module dependencies
srun python SolarSystemv2.py                    # Example Process
