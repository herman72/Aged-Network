import scipy as sc
from scipy.misc import comb
import seaborn as sns
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.patches as mpatches





def Calculate_Energy(i,j,Energy_Adj):

    Sum = 0

    for k in range(len(Energy_Adj)):

        Sum += Energy_Adj[i,k] * Energy_Adj[j,k]

    return (((-Sum) * Energy_Adj[i,j])/sc.special.comb(len(Energy_Adj),3))





def Cal_Ene_Tot(Energy_Adj,Node):
    Ene_Tot = 0

    for i in range(len(Energy_Adj)):
        for j in range(len(Energy_Adj)):
            Ene_Tot += Calculate_Energy(i, j, Energy_Adj)

    return (Ene_Tot) /6




def Mainfunc(Node,age,iterate,ensemble,std):

    Node = Node
#     Tot_Time = np.zeros((1, 20000))
#     Tot_Ene = np.zeros((1, 20000))
    Ensemble = ensemble
    age = age
    iterate = iterate
    Aging = np.ones((Node,Node),dtype=int)
#     Age_Imshow = []
#     Age_Counter = 1
    
    '''Main'''
    Age_adj_ensemble = []
    Energy_Adj_ensemble = []
    Mean_Age_ensemble = []
    Std_Age_ensenble = []
    Time_Itrate_ensemble = []

    

    #ensumble
    for ens in range(Ensemble):


        T = 0                                                 #Step
        Time = []                                             #List Step
        Energy_Adj = np.zeros((Node,Node),dtype=int)          #Energy Matrix(staet random)
        Mat_Energy = []                                       #Save Energy in every step
        Age_adj = np.zeros((Node, Node), dtype=int)           #Age Matrix
        Eold = 0
        
        
        Mean_Age = []
        Std_Age = []
        Time_Itrate = []


        
        #add Gussian age to nodes
        
        Random_gaussian = sc.stats.halfnorm.rvs(size=(Node,Node),scale=std)
        Age_gaussian = np.triu(Random_gaussian, k=1)
        Age_gaussian = Age_gaussian.round()
        Age_adj = np.copy(Age_gaussian)
        
        
#         Random_gaussian = np.random.normal(0,std,(Node,Node))
#         Age_gaussian = np.absolute(Random_gaussian - np.mean(Random_gaussian))
#         Age_gaussian = np.triu(Age_gaussian, k=1)
#         Age_adj = np.copy(Age_gaussian)

        
        
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
        
        #Age_Imshow.append(Age_adj)
        # for i in range(len(Energy_Adj)):
        #     Energy_Adj[i, i] = 0
        #     Age_adj[i, i] = 0
        #     Aging[i,i] = 0



        Eold = Cal_Ene_Tot(Energy_Adj, Node)
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
                    for j in range(Node):
                        p = 0
                        p = rnd.choice([-1, 1])

                        if Age_adj[i, j] == age:
                            Energy_Adj[i, j] = p * Energy_Adj[i, j]
                            Energy_Adj[j, i] = p * Energy_Adj[j, i]
                            Copy_Ene_mat[i, j] = Energy_Adj[i, j]
                            Copy_Ene_mat[j, i] = Energy_Adj[j, i]

                            Age_adj[i, j] = 0
                            Age_adj[j, i] = 0

                            newE = Calculate_Energy(i, j, Energy_Adj)
                            oldE = p * newE
                            dE += newE - oldE

                Eold += dE
#                 T += 1
#                 Time.append(T)
                #Mat_Energy.append(Eold)




            #Change links with energy properties

            i = rnd.randint(0, Node-1)
            j = rnd.randint(0, Node-1)

            Copy_Ene_mat[i, j] = -Copy_Ene_mat[i, j]
            Copy_Ene_mat[j, i] = -Copy_Ene_mat[j, i]

            dE = 0
            dE = -2 * Calculate_Energy(i, j, Copy_Ene_mat)
            if 0 < dE:

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
            
            
#             if t == 100 * Age_Counter:
#                 Age_Imshow.append(Age_adj)
#                 Age_Counter += 1
            Mean_Age.append(np.mean(Age_adj)/sc.special.comb(len(Energy_Adj),2))
            Std_Age.append(np.std(Age_adj)/sc.special.comb(len(Energy_Adj),2))
            Time_Itrate.append(t/sc.special.comb(len(Energy_Adj),2))
            
        Age_adj_ensemble.append(Age_adj)
        Energy_Adj_ensemble.append(Energy_Adj)


        Mean_Age_ensemble.append(Mean_Age)
        #print(Std_Age)
        Std_Age_ensenble.append(Std_Age)
        #Time_Itrate_ensemble.append(Time_Itrate)




    #np.savetxt('STD'+ str(thread_no)+str(ens) + '.txt', Std_Age_ensenble)
    #np.savetxt('LifeTime'+str(thread_no)+'.txt',Time_Itrate)
    #return Time, Mat_Energy, Age_adj,Age_Imshow,Std_Age,Mean_Age,Time_Itrate   

        
            
    
    return Time, Mat_Energy, Age_adj,Std_Age,Mean_Age,Time_Itrate


# Fun_Test = Mainfunc(64,5500,75000,1,10)

Mean_Ene = []
Thresh = []
for i in range(100,1000,50):
    Fun_Test1 = Mainfunc(32,i,15000,1,10)
    Mean_Ene.append(np.mean(Fun_Test1[1]))
    Thresh.append(i/sc.special.comb(32,2))


np.savetxt('T32.txt',Thresh)
np.savetxt('E32.txt',Mean_Ene)


    