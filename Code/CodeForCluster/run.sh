#PBS -N myjob

#PBS -l nodes=1:ppn=4

#PBS -q batch

#PBS -m abe

#PBS -M your@email.adress



cd $PBS_O_WORKDIR

python3 CodeForCluster.py
