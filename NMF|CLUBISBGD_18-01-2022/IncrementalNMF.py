import numpy as np 
import csv
import re
import sys
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
from sklearn.preprocessing import normalize
from numba import njit
from collections import defaultdict
import random
from data import ImplicitData
import numpy as np
from BISGD import BISGD
from collections import defaultdict
from eval_implicit import EvalPrequential
from datetime import datetime
import getopt
import sys


interleave = 100




def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class CLUBISGD(BISGD):
    def __init__(self, data: ImplicitData, clusters: dict, num_factors: int = 10, num_iterations: int = 10, num_nodes: int = 10, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1):
        """    Constructor.
        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        num_nodes -- Number of 'weak' learners (int, default 8)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)"""
        self.FuzzyClusters = clusters
        super().__init__(data, num_factors, num_iterations, num_nodes, learn_rate, u_regularization, i_regularization, random_seed)

    def _InitModel(self):
        super()._InitModel()
        #self.user_k = [[] for _ in range(self.num_nodes)]

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
        """
        Incrementally updates the model.
        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """

        user_id, item_id = self.data.AddFeedback(user, item)

        if len(self.user_factors[0]) == self.data.maxuserid:
            for node in range(self.num_nodes):
                #self.user_k[node].append(np.random.poisson(1, size=1)[0])
                self.user_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))
        if len(self.item_factors[0]) == self.data.maxitemid:
            for node in range(self.num_nodes):
                self.item_factors[node].append(np.random.normal(0.0, 0.1, self.num_factors))

        for node in range(self.num_nodes):
            print(node)
            print(self.num_nodes)
            print(user)
            print(self.FuzzyClusters[user])
            k = self.FuzzyClusters[user][str(node)]
            print(k)

            if k > 0.95/self.num_nodes: #higher than 95% from a random cluster allocation probability
                #self._UpdateFactors(user_id, item_id, node)
                self._UpdateFactors2(user, user_id, item_id, node) # use for weighted clustering

    def _UpdateFactors2(self, user, user_id, item_id, node, update_users: bool = True, update_items: bool = True, target: int = 1):

        p_u = self.user_factors[node][user_id]
        q_i = self.item_factors[node][item_id]

        for _ in range(int(self.num_iterations)):
            err = target - np.inner(p_u, q_i)
            err *= self.FuzzyClusters[user][str(node)]*self.num_nodes

            if update_users:
                delta = self.learn_rate * (err * q_i - self.user_regularization * p_u)
                p_u += delta

            if update_items:
                delta = self.learn_rate * (err * p_u - self.item_regularization * q_i)
                q_i += delta

        self.user_factors[node][user_id] = p_u
        self.item_factors[node][item_id] = q_i


class cluster:

    pList = []
    X = []
    name = ""

    def __init__(self,name):
        self.name = name
        self.pList = []
        self.X = []

    def addPoint(self,point):
        self.pList.append(point)
        self.X.append(point)


    def remPoint(self,point):
        #print(self.pList)
        #print(point)
        ind = self.pList.index(point)
        del self.X[ind]
        del self.pList[ind]

       
    def getPoints(self):
        return self.pList

    def erase(self):
        self.pList = []  

    def getX(self):
        return self.X

    def has(self,point):

        if point in self.pList:
            return True

        return False
    def size(self):
        return len(self.pList)    

    def printPoints(self):
        print(self.name+' Points:')
        print('-----------------')
        print(self.pList)
        print(len(self.pList))
        print('-----------------')  


def sumvector(V,x):
    result = []
    for v in V:
        temp = v + x
        result.append(temp)
    return result    

def multvector(x,V):
    result = []
    for v in V:
        temp = v * x
        result.append(temp)
    return result    

def TransformVec(V):
    new = normalize(V[:,np.newaxis], axis=0).ravel()
    for i in range(len(V)):
        V[i] = sigmoid(V[i])
    new = normalize(V[:,np.newaxis], axis=0).ravel()    

    return new

sys.path.append('./')
configPath = 'config'
dataPath = 'ppf2.csv'

Data = []
with open(dataPath, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        Data.append([float(row[0]),float(row[1])])

feat = 10
iters = 10

lamb = 0.1
eta = 1
User_register =[]
Item_register =[]
A = []
B = []
Clusters = []
Existing_Clusters = []
Dic = defaultdict(dict)
unchanged_Dic = defaultdict(dict)
for d in Data:
    if d[0] not in User_register:
        User_register.append(d[0])
        A_u = np.random.normal(0.0, 0.1, size = feat)
 
    if d[1] not in Item_register:
        Item_register.append(d[1])
        B_i = np.random.normal(0.0, 0.1, size = feat)


    for x in range(iters):
        err = 1 - np.dot(A_u, np.transpose(B_i))
        
        #err = float(err)
        #print(err)
        auxA = np.multiply(B_i,err)
        auxB = np.multiply(A_u,err)
        A_u = np.add(A_u, np.subtract(np.multiply(auxA,eta), np.multiply(A_u,lamb)))
        #print(A_u)
        B_i = np.add(B_i, np.subtract(np.multiply(auxB,eta), np.multiply(B_i,lamb)))
        print(err)

    #print(err)
    unchanged_A_u = A_u
    A_u = TransformVec(A_u)
    B_i = TransformVec(B_i)
    A.append(A_u)
    B.append(B_i)

    for f in range(len(A_u)):
        Dic[d[0]][str(f)]= A_u[f]        
        #cluster_name = 'Cluster' + str(f)        
        # if(cluster_name not in Existing_Clusters):
        #     C = cluster(cluster_name)
        #     C.addPoint(d[0])
        #     Existing_Clusters.append(cluster_name)
        #     Clusters.append(C)
        # else:
        #     for clust in Clusters:
        #         if clust.name == cluster_name:
        #             clust.addPoint(f)    
                      
                
#print(Dic)
#print(Dic[22808.0])
Data_Transpose = np.transpose(Data)
stream = ImplicitData(Data_Transpose[0],Data_Transpose[1])
model = CLUBISGD(stream, Dic)

eval = EvalPrequential(model,stream, metrics = ["Recall@20"])

start_recommend = datetime.now()
print('start time', start_recommend)

results=eval.Evaluate()

print('npmean(resuls[Recall@20])', np.mean(results['Recall@20']))

end_recommend = datetime.now()
print('end time', end_recommend)

tempo = end_recommend - start_recommend

print('run time', tempo)
print('')
print('get tuple',np.mean(results['time_get_tuple']))
print('recommend',np.mean(results['time_recommend']))
print('eval_point',np.mean(results['time_eval_point']))
print('update',np.mean(results['time_update']))