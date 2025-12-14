#!/bin/bash
#SBATCH --job-name=f05-24-120
#SBATCH --nodes=2
#SBATCH --time=2-00:00:00
#-SBATCH --account=
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ebranlard@umass.edu
#SBATCH -o slurm-%x-%j.log
#-SBATCH --qos=high
#-SBATCH -p batch

# --------------------- INPUT ----------------------------
ENV=exawind-cpu
nalu_exec=naluX
nalu_inputs=("input.yaml")
EXAWIND_MANAGER=/scratch/%USERNAME%/exawind-manager

# ------------------- MODULES ----------------------------
module purge
module load PrgEnv-intel
module load cray-python 

# ------------- SETUP EXAWIND MANAGER -------------------
source ${EXAWIND_MANAGER}/start.sh && spack-start 
echo "# >>> Activating environment  : ${ENV}"
spack env activate -d "${EXAWIND_MANAGER}/environments/${ENV}"  || exit 1
spack load exawind

ranks_per_node=104
mlupi_ranks=$(expr $SLURM_JOB_NUM_NODES \* $ranks_per_node)
export OMP_NUM_THREADS=1  # Max hardware threads = 4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

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
    srun -u -N$SLURM_NNODES -n384 --ntasks-per-node=96 --distribution=block:cyclic --cpu_bind=cores \
        ${nalu_exec} -i ${nalu_input}
    echo "#>>> Done         on: $(date)"
    echo "------------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------------"
done
echo "#>>> Ending job      on:  $(date)"

#srun -u -N3 -n312 --ntasks-per-node=104 --distribution=cyclic:cyclic --cpu_bind=cores ${nalu_exec} -i ${nalu_input} 
#srun -u -N6 -n312 --ntasks-per-node=52 --distribution=cyclic:cyclic --cpu_bind=cores ${nalu_exec} -i ffa_w3_211_static_aoa_30.yaml -o log.out 
#--- Shreyas
#srun -u -N4 -n384 --ntasks-per-node=96 --distribution=block:cyclic --cpu_bind=map_cpu:0,52,13,65,26,78,39,91,1,53,14,66,27,79,40,92,2,54,15,67,28,80,41,93,3,55,16,68,29,81,42,94,4,56,17,69,30,82,43,95,5,57,18,70,31,83,44,96,6,58,19,71,32,84,45,97,7,59,20,72,33,85,46,98,8,60,21,73,34,86,47,99,9,61,22,74,35,87,48,100,10,62,23,75,36,88,49,101,11,63,24,76,37,89,50,102,12,64,25,77,38,90,51,103 ${nalu_exec} -i $grids${list_of_cases[$idx]}/ffa_w3_211_static_${list_of_cases[$idx]}.yaml -o $grids${list_of_cases[$idx]}/log$idx.out &

#srun -u -N6 -n312 --ntasks-per-node=52 --distribution=cyclic:cyclic --cpu_bind=cores ${nalu_exec} -i $grids${list_of_cases[$idx]}/*.yaml -o $grids${list_of_cases[$idx]}/log$idx.out &

# Adjust the ratio of total MPI ranks for AMR-Wind and Nalu-Wind as needed by a job 
# srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) \
# --distribution=block:block --cpu_bind=rank_ldom exawind --awind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.25) \
# --nwind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.75) <input-name>.yaml
