#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=2
#SBATCH --time=2-00:00:00
#-SBATCH --account=
#-SBATCH --mail-type=ALL
#-SBATCH --mail-user=
#-SBATCH -o slurm-%x-%j.log
#-SBATCH --qos=high

nalu_exec=naluX
nalu_input=input.yaml

#module purge
#module load PrgEnv-intel
#module load cray-python 
#
#export EXAWIND_MANAGER=/scratch/XX/exawind-manager
#source ${EXAWIND_MANAGER}/start.sh && spack-start 
#spack env activate -d ${EXAWIND_MANAGER}/environments/exawind-cpu
#spack load exawind

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


echo "#>>> Starting NALU  = -N  -n ${SLURM_NTASKS}   ${nalu_exec} ${nalu_input}"
mpiexec -n ${SLURM_NTASKS} ${nalu_exec} -i ${nalu_input} 
#srun -u -N3 -n{SLURM_NTASKS} ${nalu_exec} -i ${nalu_input} 
echo "Done"
