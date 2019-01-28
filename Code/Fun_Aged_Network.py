## Created By: Mohammad Sherafati(Herman), 28Jan 2019, m.sherafati7@gmail.com

#'''Import Package'''
import scipy as sc



#'''Var'''



#'''Main Code '''


# Calculate A triangle
def Calculate_Energy(i,j,Energy_Adj):

    Sum = 0

    for k in range(len(Energy_Adj)):

        Sum += Energy_Adj[i,k] * Energy_Adj[j,k]

    return (((-Sum) * Energy_Adj[i,j])/sc.special.comb(len(Energy_Adj),3))






# Calculate total Energy
def Cal_Ene_Tot(Energy_Adj,Node):
    Ene_Tot = 0

    for i in range(len(Energy_Adj)):
        for j in range(len(Energy_Adj)):
            Ene_Tot += Calculate_Energy(i, j, Energy_Adj)

    return (Ene_Tot) /6








