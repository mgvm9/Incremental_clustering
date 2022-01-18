from data import ImplicitData
from recommenders_implicit import *
import numpy as np
import pandas as pd
import time
#import metrics

class EvalPrequential:

    def __init__(self, model: Model, data: ImplicitData, metrics = ["Recall@20"]):
        # TODO: Input checks
        self.model = model
        self.data = data
        self.metrics = metrics

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
        results = dict()

        if not count:
            count = self.data.size

        count = min(count, self.data.size - start)

        for metric in self.metrics:
            results[metric] = np.zeros(count)

        for i in range(count):
            uid, iid = self.data.GetTuple(i + start)
            reclist = self.model.Recommend(uid)
            #print('reclist', reclist)
            results[metric][i] = self.__EvalPoint(iid, reclist)
            #print('results', results)
            self.model.IncrTrain(uid, iid)

        return results

    def __EvalPoint(self, item_id, reclist):
        result = 0
        for metric in self.metrics:
            if metric == "Recall@20":
                #print('reclist', reclist)
                #print('len(reclist)', len(reclist))
                reclist = [x[0] for x in reclist[:20]]
                result = int(item_id in reclist)
        return result
