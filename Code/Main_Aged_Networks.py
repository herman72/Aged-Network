## Created By: Mohammad Sherafati, 2Dec 2018, m.sherafati7@gmail.com

#  '''Import Package'''

import scipy as sc
from scipy.misc import comb
import seaborn as sns
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.patches as mpatches
import Fun_Aged_Network as Fun_Aged_Network
import multiprocessing as mp
import os.path

# '''Variable'''

Path = '/home/mohammad/Documents/Thesis/Aged-Network/Code'

#Set_Age = 100
#ensemble = 1

# '''Main Function'''

def Mainfunc(thread_no,Node,age,iterate,ensemble,std):

    Node = Node
    Ensemble = ensemble
    age = age
    iterate = iterate
    Aging = np.ones((Node,Node),dtype=int)


    # Age_Imshow = []
    # Age_Counter = 1

    '''Main'''

    Age_adj_ensemble = []
    Energy_Adj_ensemble = []
    Mean_Age_ensemble = []
    Std_Age_ensenble = []
    Time_Itrate_ensemble = []


    #ensumble

    for ens in range(Ensemble):
        #print(ens)

        T = 0                                                 #Step
        Time = []                                             #List Step
        Energy_Adj = np.zeros((Node,Node),dtype=int)          #Energy Matrix(staet random)
        Mat_Energy = []                                       #Save Energy in every step
        Age_adj = np.zeros((Node, Node), dtype=int)           #Age Matrix
        Eold = 0

        #Age_Imshow = []




        Mean_Age = []
        Std_Age = []
        Time_Itrate = []
        
        
        #add Gussian age to nodes

        Random_gaussian = sc.stats.halfnorm.rvs(size=(Node,Node),scale=std)
        Age_gaussian = np.triu(Random_gaussian, k=1)
        Age_gaussian = Age_gaussian.round()
        Age_adj = np.copy(Age_gaussian)
        
        
        # Random_gaussian = np.random.normal(0,std,(Node,Node))
        # Age_gaussian = np.absolute(Random_gaussian - np.mean(Random_gaussian))
        # Age_gaussian = np.triu(Age_gaussian, k=1)
        # Age_adj = np.copy(Age_gaussian)

        
        
        #Create Random First State

        for i in range(len(Energy_Adj)):
            
            for j in range(i,len(Energy_Adj)):
                Energy_Adj[i, j] = rnd.choice([-1, 1])
                Energy_Adj[j, i] = Energy_Adj[i, j]

                Age_adj[j,i] = Age_adj[i,j]
    

        #Zero Diognal

        np.fill_diagonal(Energy_Adj,0)
        np.fill_diagonal(Age_adj,0)
        np.fill_diagonal(Aging, 0)
        Age_adj = Age_adj.round().astype(int)
        



        Eold = Fun_Aged_Network.Cal_Ene_Tot(Energy_Adj, Node)
        Copy_Ene_mat = np.copy(Energy_Adj)



    #def func(Node,iterate,Energy_Adj):



        for t in range(iterate):

            Age_network =0
            dE = 0
            Age_network = Age_adj.max()                       #Find Oldest link


            #Change old links with random links
            if Age_network == age:
                dE = 0
                for i in range(Node):
                    for j in range(i,Node):
                        p = 0
                        p = rnd.choice([-1, 1])

                        if Age_adj[i, j] == age:
                            Energy_Adj[i, j] = p * Energy_Adj[i, j]
                            Energy_Adj[j, i] = p * Energy_Adj[j, i]
                            Copy_Ene_mat[i, j] = Energy_Adj[i, j]
                            Copy_Ene_mat[j, i] = Energy_Adj[j, i]

                            Age_adj[i, j] = 0
                            Age_adj[j, i] = 0

                            newE = Fun_Aged_Network.Calculate_Energy(i, j, Energy_Adj)
                            oldE = p * newE #Calculate_Energy(i, j, Energy_Adj)
                            dE += newE - oldE

                Eold += dE
                # T += 1
                # Time.append(T)
                Mat_Energy.append(Eold)




            #Change links with energy properties

            i = rnd.randint(0, Node-1)
            j = rnd.randint(0, Node-1)

            Copy_Ene_mat[i, j] = -Copy_Ene_mat[i, j]
            Copy_Ene_mat[j, i] = -Copy_Ene_mat[j, i]

            dE = 0
            dE = -2 * Fun_Aged_Network.Calculate_Energy(i, j, Copy_Ene_mat)
            if 0 <= dE:

                Energy_Adj[i, j] = -Energy_Adj[i, j]
                Energy_Adj[j, i] = -Energy_Adj[j, i]

                Eold = Eold - dE

                Mat_Energy.append(Eold)
                T +=1
                Time.append(T)

                Age_adj = Age_adj + Aging
                Age_adj[i, j] = 0
                Age_adj[j, i] = 0

            else:
                Copy_Ene_mat[i, j] = Energy_Adj[i, j]
                Copy_Ene_mat[j, i] = Energy_Adj[j, i]
                T += 1
                Time.append(T)
                Mat_Energy.append(Eold)
                #Age_adj = Age_adj + Aging

            # if t == 100 * Age_Counter:
            #     Age_Imshow.append(Age_adj)
            #     Age_Counter += 1

            Mean_Age.append(np.mean(Age_adj)/sc.special.comb(len(Energy_Adj),2))
            Std_Age.append(np.std(Age_adj)/sc.special.comb(len(Energy_Adj),2))
            #print(Std_Age)

            Time_Itrate.append(t/sc.special.comb(len(Energy_Adj),2))

        Age_adj_ensemble.append(Age_adj)
        Energy_Adj_ensemble.append(Energy_Adj)


        Mean_Age_ensemble.append(Mean_Age)
        #print(Std_Age)
        Std_Age_ensenble.append(Std_Age)
        #Time_Itrate_ensemble.append(Time_Itrate)




    np.savetxt('/home/mohammad/Documents/Thesis/Aged-Network/Code/Output100/'+'STD100'+ str(thread_no)+str(ens) + '.txt', Std_Age_ensenble)
    #np.savetxt('LifeTime'+str(thread_no)+'.txt',Time_Itrate)
    #return Time, Mat_Energy, Age_adj,Age_Imshow,Std_Age,Mean_Age,Time_Itrate   



# Fun_Test = Mainfunc(32,10000,6000,1,10)
# plt.plot(Fun_Test[0],Fun_Test[1])
# plt.show()


Node = 32
age = 100
iterate = 6000
ensemble = 1
std = 10

if __name__ == "__main__":
    thread_no = 0
    for i in range(4):
        processes = []
        for j in range(4):
            processes.append(mp.Process(target=Mainfunc, args=(thread_no,Node,age,iterate,ensemble,std)))
            thread_no += 1
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    # Time = np.arange(1, eter+1, 1)
    # np.savetxt('Time.txt',np.log(Time) )


import time

start = time.time()
print("hello")
end = time.time()
print(end - start)
