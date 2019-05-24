#PBS -N myjob

#PBS -m abe

#PBS -M your@email.adress

#PBS -l nodes=1:ppn=4

cd $PBS_O_WORKDIR

python3 CodeForCluster64.py
