import numpy as np 
import csv
import re
import sys
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
from sklearn.preprocessing import normalize

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
    return new

sys.path.append('./')
configPath = 'config'
dataPath = 'ppf2.csv'

Data = []
with open(dataPath, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        Data.append([float(row[0]),float(row[1])])

feat = 5
iters = 1000

lamb = 0.1
eta = 1
User_register =[]
Item_register =[]
A = []
B = []
Clusters = []
Existing_Clusters = []

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

    A_u = TransformVec(A_u)
    B_i = TransformVec(B_i)
    A.append(A_u)
    B.append(B_i)

for f in range(len(A)):
    biggestValue = 0
    feature_cluster = 0
    i = A[f]
    for j in range(feat):
        if i[j] > biggestValue:
            biggestValue = i[j]
            feature_cluster = j
    cluster_name = 'Cluster' + str(feature_cluster)        
    if(cluster_name not in Existing_Clusters):
        C = cluster(cluster_name)
        C.addPoint(i)
        Existing_Clusters.append(cluster_name)
        Clusters.append(C)
    else:
        for clust in Clusters:
            if clust.name == cluster_name:
                clust.addPoint(f)
            
for clust in Clusters:
    clust.printPoints()


