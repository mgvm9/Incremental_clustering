from data import ImplicitData
from param_tuning import PatternSearchISGD
from pprint import pprint
import pandas as pd
from operator import itemgetter
import numpy as np
from recommenders_implicit import * #BISGD
#from recommenders_implicit.BISGD_ import BISGD
from recommenders_implicit.UBISGD import UBISGD
from eval_implicit.EvalPrequential2_ import EvalPrequential
from datetime import datetime

data = pd.read_csv("datasets/playlisted_tracks.tsv","\t")
stream = ImplicitData(data['playlist_id'],data['track_id'])

numeroNodes = 2
#model = BISGD(ImplicitData([],[]),200,6, numeroNodes, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5, use_numba = False)
model = UBISGD(ImplicitData([],[]),200,6, numeroNodes, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5, use_numba = False)

eval = EvalPrequential(model,stream, metrics = ["Recall@20"])

start_recommend = datetime.now()
print('start time', start_recommend)

resultados=eval.Evaluate(0,stream.size)

print('sum(resultados[Recall@20])/stream.size', sum(resultados['Recall@20'])/stream.size)

##BISGD, recall@20
#1 node
#0.18298
#runtime 2:13:07   
#recall20 n=2
# 0.1951
#run time 3:40:22
#6 nodes 
#recall20 = 0.2022
#run time = 9:09:32
#8 nodes
#recall20= 0.2040
# run time 12:16:38
#12 nodes
#recall20= 0.20316
# run time 17:58:27

#recall20 ISGD= 0.256

#UBISGD, recall@20
#1 node
# 0.1468
#run time 2:16:37

#2 nodes
#0.2033
#run time 3:30:18

#6 nodes
#0.2520
#run time 9:08:43

#8 nodes
#0.2561
#run time 12:09:41

#12 nodes
#0.260
#run time 17:44:09

#16 nodes
#0.2639
#run time 23:45:49

#24 nodes
#0.2634
#run time 1 day, 23:03:26.809245

end_recommend = datetime.now()
print('end time', end_recommend)

tempo = end_recommend - start_recommend

print('run time', tempo)
