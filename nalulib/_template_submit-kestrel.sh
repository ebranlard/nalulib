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

nalu_exec=naluX
nalu_input=input_AoA0_n120.yaml

module purge
module load PrgEnv-intel
module load cray-python 

export EXAWIND_MANAGER=/scratch/ebranlar/exawind-manager
source ${EXAWIND_MANAGER}/start.sh && spack-start 
spack env activate -d ${EXAWIND_MANAGER}/environments/exawind-cpu
spack load exawind

ranks_per_node=104
mlupi_ranks=$(expr $SLURM_JOB_NUM_NODES \* $ranks_per_node)
export OMP_NUM_THREADS=1  # Max hardware threads = 4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# ---- DEBUG
echo ">>> Job name       = $SLURM_JOB_NAME"
echo ">>> Job ID         = $SLURM_JOBID"
echo ">>> Num. nodes     = $SLURM_JOB_NUM_NODES"
echo ">>> Task p node    = $SLURM_NTASKS_PER_NODE"
#echo ">>> N should be    = $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))"
echo ">>> Num. MPI Ranks = $mpi_ranks"
echo ">>> Num. threads   = $OMP_NUM_THREADS"
echo ">>> Working dir    = $PWD"
echo ">>> Date           = `date`"
echo ">>> Directory content:"
ls -alh
echo ">>> module list    ="
module list
# ---- END DEBUG

echo ">>> Starting NALU  = ${nalu_exec} ${nalu_input}"
#srun -u -N3 -n312 --ntasks-per-node=104 --distribution=cyclic:cyclic --cpu_bind=cores ${nalu_exec} -i ${nalu_input} 
srun -u -N2 -n192 --ntasks-per-node=96 --distribution=block:cyclic --cpu_bind=cores ${nalu_exec} -i input_AoA0_n120.yaml -o input_AoA0_n120.log 
srun -u -N2 -n192 --ntasks-per-node=96 --distribution=block:cyclic --cpu_bind=cores ${nalu_exec} -i input_AoA0_n24.yaml  -o input_AoA0_n24.log 
srun -u -N2 -n192 --ntasks-per-node=96 --distribution=block:cyclic --cpu_bind=cores ${nalu_exec} -i input_AoA5_n120.yaml -o input_AoA5_n120.log 
srun -u -N2 -n192 --ntasks-per-node=96 --distribution=block:cyclic --cpu_bind=cores ${nalu_exec} -i input_AoA5_n24.yaml  -o input_AoA5_n24.log 
echo "Done"

#--- Shreyas
#srun -u -N4 -n384 --ntasks-per-node=96 --distribution=block:cyclic --cpu_bind=map_cpu:0,52,13,65,26,78,39,91,1,53,14,66,27,79,40,92,2,54,15,67,28,80,41,93,3,55,16,68,29,81,42,94,4,56,17,69,30,82,43,95,5,57,18,70,31,83,44,96,6,58,19,71,32,84,45,97,7,59,20,72,33,85,46,98,8,60,21,73,34,86,47,99,9,61,22,74,35,87,48,100,10,62,23,75,36,88,49,101,11,63,24,76,37,89,50,102,12,64,25,77,38,90,51,103 ${nalu_exec} -i $grids${list_of_cases[$idx]}/ffa_w3_211_static_${list_of_cases[$idx]}.yaml -o $grids${list_of_cases[$idx]}/log$idx.out &
