from data import ImplicitData
from collections import defaultdict
from recommenders_implicit import *
import numpy as np
import pandas as pd
import time


class EvalPrequential2:

    def __init__(self, model: Model, data: ImplicitData, metrics = ["Recall@20"], NrNodes: int = 5):
        # TODO: Input checks
        self.model = model
        self.data = data
        self.metrics = metrics
        self.nrNodes= NrNodes


    def EvaluateTime(self, start = 0, count = 0):
        results = dict()
        time_get_tuple = np.zeros(self.data.size)
        time_recommend = np.zeros(self.data.size)
        time_eval_point = np.zeros(self.data.size)
        time_update = np.zeros(self.data.size)

        if not count:
            count = self.data.size
        for metric in self.metrics:
            results[metric] = np.zeros(count)
        for i in range(count):
            if i % (count/100) == 0:
                print(".", end = '', flush = True)

            start_get_tuple = time.time()
            uid, iid = self.data.GetTuple(i + start)
            end_get_tuple = time.time()
            time_get_tuple[i] = end_get_tuple - start_get_tuple

            start_recommend = time.time()
            reclist = self.model.Recommend(uid)
            end_recommend = time.time()
            time_recommend[i] = end_recommend - start_recommend

            start_eval_point = time.time()
            results[metric][i] = self.__EvalPoint(iid, reclist)
            end_eval_point = time.time()
            time_eval_point[i] = end_eval_point - start_eval_point

            start_update = time.time()
            self.model.IncrTrain(uid, iid)
            end_update = time.time()
            time_update[i] = end_update - start_update

        results['time_get_tuple'] = time_get_tuple
        results['time_recommend'] = time_recommend
        results['time_eval_point'] = time_eval_point
        results['time_update'] = time_update

        return results

    def Evaluate(self, start = 0, count = 0):
        results = defaultdict(dict)
        reclist = {}

        entra_uid = {}
        entra_iid = {}

        if not count:
            count = self.data.size

        count = min(count, self.data.size - start)

        for i in range(count):
            uid, iid = self.data.GetTuple(i + start)
            entra_uid[uid] = 0
            entra_iid[iid] = 0

        for metric in self.metrics:
            for node in range(self.nrNodes):
                results[metric][node] = np.zeros(count)


        for i in range(count):
            uid, iid = self.data.GetTuple(i + start)            
            for node in range(self.nrNodes):
                reclist[node] = self.model.Recommend(uid, node, entra_uid[uid])
                results[metric][node][i] = self.__EvalPoint(iid, reclist[node])
                self.model.IncrTrain(uid, iid, node, entra_uid[uid], entra_iid[iid])
                entra_uid[uid] += 1
                entra_iid[iid] += 1

        return results

    def __EvalPoint(self, item_id, reclist):
        result = 0
        for metric in self.metrics:
            if metric == "Recall@20":
                reclist= [x[0] for x in reclist[:20]]
                result = int(item_id in reclist)                
        return result
