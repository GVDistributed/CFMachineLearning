#!/usr/bin/python

import sys
import cPickle
import logging
import random
import itertools
from collections import defaultdict

from numpy import *

from baseline import GroupLensDataSet
from baseline import CFModel as CFModelBase

from utils import binary_search
from utils import WindowedAverage

import matplotlib.pyplot as plt

class CFModel(CFModelBase):

    def cbui(self, user_id, item_id, t):
        return self.mu.query(t) + self.cbu[user_id].query(t) + self.cbi[item_id].query(t)

    def bui(self, user_id, item_id, t):
        return self.mu.query(t) + self.bu[user_id] + self.bi[item_id]

    def rui(self, user_id, item_id, ratings, t):
        p = len(ratings)**(-self.alpha) * \
            sum((r - self.cbui(user_id, item_id, t))*self.x[item_id] for item_id, r, t in ratings)

        return self.cbui(user_id, item_id, t) + dot(self.q[item_id], p)

    def train(self, data, reg, reg_i, reg_u, min_iter, max_iter, step_size):

        logging.info("Computing mu...")  
        self.mu = WindowedAverage(5000)
        for v1, v2, r, t in data.iter_ratings():
            self.mu.add(t, r)
        self.mu.process()

        logging.info("Computing item baselines...")
        self.cbi = []
        for ratings in data.r_i:
            cb = WindowedAverage(2000, reg_i)
            for user_id, r, t in ratings:
                cb.add(t, r - self.mu.query(t))
            self.cbi.append(cb.process())

        logging.info("Computing user baselines...")
        self.cbu = []
        for user_id, ratings in data.iter_users():
            cb = WindowedAverage(1000, reg_u)
            for item_id, r, t in ratings:
                cb.add(t, r - self.cbi[item_id].query(t) - self.mu.query(t))
            self.cbu.append(cb.process())

        logging.info("Performing optimization...")
        self.bi = array([x.query(0) for x in self.cbi])
        self.bu = array([x.query(0) for x in self.cbu])
        if self.x is None:
            self.x = [array([(random.random()-0.5)/100000.0 \
                for i in range(self.f)]) for j in range(data.m)]
        if self.q is None:
            self.q = [array([(random.random()-0.5)/100000.0 \
                for i in range(self.f)]) for j in range(data.m)]

        last_tot = float('inf')
        for iter in range(max_iter):
            tot = 0
            rmse = 0
            n = 0
            for user_id in range(data.n):
                p = len(data.r_u[user_id])**(-self.alpha) * \
                    sum((r - self.cbui(user_id, item_id, t))*self.x[item_id] \
                        for item_id, r, t in data.r_u[user_id])

                s = 0
                for item_id, r, t in data.r_u[user_id]:
                    rp = self.bui(user_id, item_id, t) + dot(self.q[item_id], p)
                    e = r - rp

                    tot += e**2 + reg*(self.bu[user_id]**2 + self.bi[item_id]**2 + \
                        dot(self.q[item_id], self.q[item_id]))

                    s += e*self.q[item_id]
                    self.q[item_id] += step_size * (e*p - reg*self.q[item_id])
                    self.bu[user_id] += step_size * (e - reg*self.bu[user_id])
                    self.bi[item_id] += step_size * (e - reg*self.bi[item_id]) 

                    rmse += e**2
                    n += 1
 
                for item_id, r, t in data.r_u[user_id]:
                    tot += reg * len(data.r_u[user_id]) * \
                        dot(self.x[item_id], self.x[item_id])

                    self.x[item_id] += step_size * \
                        (len(data.r_u[user_id])**(-self.alpha) * \
                            (r - self.cbui(user_id, item_id, t))*s - reg*self.x[item_id])

            logging.info("%s: %s, %s", iter, (rmse / n) ** 0.5, tot)

            if iter >= min_iter and tot > last_tot:
                logging.info("Stopping early")
                return

            last_tot = tot

def validate(model, train, test, reg, reg_i, reg_u, 
             min_iter=10, max_iter=100, step_size=0.01, save=True):

    model.train(train, reg, reg_i, reg_u, min_iter, max_iter, step_size)

    tot = 0
    n = 0
    for user_id, item_id, r, t in test.iter_ratings(train):
        rp = model.rui(user_id, item_id, train.r_u[user_id], t)
        tot += (r - rp)**2
        n += 1
    rmse = (tot / float(n)) ** 0.5

    if save:
        model.save("model[%s-%s-%s-%s].dump" % (rmse, reg, reg_i, reg_u))

    return rmse

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ######
    # 5-fold cross validation of ml-100k
    ######

    avg_rmse = 0
    for k in range(1, 6):
        train = GroupLensDataSet("ml-100k/u%s.base"%k, "\t")
        test = GroupLensDataSet("ml-100k/u%s.test"%k, "\t")
        model = CFModel()
        rmse = validate(model, train, test, reg=0.0025, reg_i=15, reg_u=25, save=False)
        model.save("model.100k-%s[%s].dump" % (k, rmse))

        '''
        fig = plt.figure()
        p = fig.add_subplot(1, 1, 1)
        p.scatter(*model.mu.to_plot(100))
        fig.show()
        fig = plt.figure()
        p = fig.add_subplot(1, 1, 1)
        p.scatter(*model.mu.to_plot(1000))
        fig.show()
        fig = plt.figure()
        p = fig.add_subplot(1, 1, 1)
        p.scatter(*model.mu.to_plot(5000))
        fig.show()
        '''

        avg_rmse += rmse / 5.0
        print k, rmse
    print avg_rmse
    
