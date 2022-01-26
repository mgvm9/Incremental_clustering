from data import ImplicitData
from pprint import pprint
import pandas as pd
from operator import itemgetter
import numpy as np
from recommenders_implicit import ISGD
from eval_implicit import EvalPrequential
from datetime import datetime

data = pd.read_csv("ppf2.csv","\t")
stream = ImplicitData(data[0],data[1])

model = ISGD(ImplicitData([],[]),200,6, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5, use_numba = False)

eval = EvalPrequential(model,stream)
"""
res = eval.EvaluateTime(0, 20000)

print("GetTuple: " + str(sum(res['time_get_tuple'])))
print("Recommend: " + str(sum(res['time_recommend'])))
print("EvalPoint: " + str(sum(res['time_eval_point'])))
print("Update: " + str(sum(res['time_update'])))
"""

start_recommend = datetime.now()
print('start time', start_recommend)

resultados3=eval.Evaluate(0,stream.size)

print(sum(resultados3['Recall@20'])/stream.size) # com hyperparametros otimizados do artigo e o dataset todo deu 0.200 de recall@10
# tempo de execução 42 min
#Recall@20=0.256 (maior que o recall10) no artigo é 0.302
end_recommend = datetime.now()
print('end time', end_recommend)

tempo = end_recommend - start_recommend

print('run time', tempo)
