#!/bin/bash
#SBATCH --job-name=FF-ST-3D
#SBATCH --time=2-00:00:00  # Job time limit Days-Hours:Minutes:Seconds
#SBATCH --nodes=1
#SBATCH --ntasks=128           # Number of MPI processes
#SBATCH --mem=300G            # Memory
#SBATCH -p cpu
#SBATCH --exclude=cpu024  
#-SBATCH --nodelist=cpu069,cpu070,cpu071,cpu072,cpu073,cpu074,cpu075,cpu076,cpu077,cpu078
#-SBATCH --exclusive  # Request entire nodes
#-SBATCH --cpus-per-task=1      # Number of Cores per Task
#-SBATCH --ntasks-per-node=24   # 
#-SBATCH --constraint=ib # for infiniband
#SBATCH --mail-user=ebranlard@umass.edu
#SBATCH --mail-type ALL # Send e-mail when job begins, ends or fails
#SBATCH --output=slurm-%x.log   # Output %j: job number, %x: jobname
#-SBATCH -G 1  # Number of GPUs
#-SBATCH -p gpu  # Partition
#-SBATCH --time=0-36
#-SBATCH --account=isda

# --------------------- INPUT ----------------------------
ENV=nalu-wind-shared
ENV=nalu-wind-nomod
nalu_exec=naluX
nalu_inputs=("input.yaml")
EXAWIND_MANAGER=/work/pi_ebranlard_umass_edu/exawind-manager

# ------------------- MODULES ----------------------------
#module purge
#module load mpich/4.2.1
#module load python/3.12.3

# ------------- SETUP EXAWIND MANAGER -------------------
echo "# >>> Activating spack from exawind-manager"
source "${EXAWIND_MANAGER}/start.sh" && spack-start
echo "# >>> Activating environment  : ${ENV}"
spack env activate -d "${EXAWIND_MANAGER}/environments/${ENV}"  || exit 1
spack load nalu-wind

#export OMP_NUM_THREADS=1  # Max hardware threads = 4
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

# ---- DEBUG
echo "#>>> JOB_NAME        = $SLURM_JOB_NAME"
echo "#>>> JOBID           = $SLURM_JOBID"
echo "#>>> JOB_NUM_NODES   = $SLURM_JOB_NUM_NODES"
echo "#>>> NNODES          = $SLURM_NNODES"
echo "#>>> NTASKS          = $SLURM_NTASKS   $SLURM_NPROCS"
echo "#>>> NTASKS_PER_CORE = $SLURM_NTASKS_PER_CORE"
echo "#>>> TASKS_PER_NODE  = $SLURM_TASKS_PER_NODE"
echo "#>>> MEM_PER_NODE    = $SLURM_MEM_PER_NODE"
echo "#>>> JOB_NODELIST    = $SLURM_JOB_NODELIST"
echo "#>>> Num. MPI Ranks  = $mpi_ranks"
echo "#>>> Num. threads    = $OMP_NUM_THREADS"
echo "#>>> Working dir     = $PWD"
echo "#>>> Date            = `date`"
echo "#>>> Working directory $SLURM_SUBMIT_DIR"
echo "#>>> Directory content:"
ls -alh
echo "#>>> module list    ="
module list
# ---- END DEBUG
for nalu_input in "${nalu_inputs[@]}"; do
    echo "------------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------------"
    echo "#>>> Starting NALU  =  -n ${SLURM_NTASKS}   ${nalu_exec} ${nalu_input}"
    echo "#>>>              on: $(date)"
    mpiexec -n ${SLURM_NTASKS}  ${nalu_exec} -i ${nalu_input}
    echo "#>>> Done         on: $(date)"
    echo "------------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------------"
done
echo "#>>> Ending job      on:  $(date)"