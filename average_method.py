#!/usr/bin/python

import sys
import cPickle
import logging
import itertools
from collections import defaultdict

from numpy import *

import random

from baseline import GroupLensDataSet
from baseline import CFModel as CFModelBase
from baseline import validate
from baseline import full_optimization

from utils import binary_search
from utils import WindowedAverage

import matplotlib.pyplot as plt

class CFModel(CFModelBase):

    def cbui(self, user_id, item_id, t):
        return self.mu.query(t) + self.cbu[user_id] + self.cbi[item_id] + self.cbit[item_id].query(t)

    def bui(self, user_id, item_id, t):
        return self.mu.query(t) + self.bu[user_id] + self.bi[item_id] + self.cbit[item_id].query(t)

    def load(self, filename):
        with open(filename) as file_read:
            self.mu, self.bu, self.bi, \
                self.cbu, self.cbi, self.cbit, self.q, self.x, \
                self.alpha, self.f = cPickle.loads(file_read.read())
        return self

    def save(self, filename):
        with open(filename, "w") as file_write:
            file_write.write(cPickle.dumps((self.mu, self.bu, self.bi, self.cbu, self.cbi, self.cbit, self.q, self.x, self.alpha, self.f), protocol=-1))

    def baselines(self, data, reg_i, reg_u, reg_it, width_mu, width_it):
        logging.info("Computing mu...")  
        self.mu = WindowedAverage(width_mu)
        for v1, v2, r, t in data.iter_ratings():
            self.mu.add(t, r)
        self.mu.process()

        logging.info("Computing item baselines...")
        self.cbi = []
        for ratings in data.r_i:
            t = 0
            n = 0
            for user_id, r, timestamp in ratings:
                t += (r - self.mu.query(timestamp))
                n += 1
            self.cbi.append(t / float(reg_i + n))
        self.cbi = array(self.cbi)

        logging.info("Computing item baseline functions...")
        self.cbit = []
        for item_id, ratings in enumerate(data.r_i):
            cb = WindowedAverage(width_it, reg_it)
            for user_id, r, t in ratings:
                cb.add(t, r - self.cbi[item_id] - self.mu.query(t))
            self.cbit.append(cb.process())

        logging.info("Computing user baselines...")
        self.cbu = []
        for user_id, ratings in data.iter_users():
            t = 0
            n = 0
            for item_id, r, timestamp in ratings:
                t += (r - self.cbi[item_id] - self.cbit[item_id].query(timestamp) - self.mu.query(timestamp))
                n += 1
            self.cbu.append(t / float(reg_u + n))
        self.cbu = array(self.cbu)

        self.bi = array(self.cbi)
        self.bu = array(self.cbu)

    def train(self, data, reg, reg_i, reg_u, reg_it, width_mu, width_it, min_iter, max_iter, step_size):
        self.baselines(data, reg_i, reg_u, reg_it, width_mu, width_it)

        logging.info("Performing optimization...")
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
                cw = len(data.r_u[user_id])**(-self.alpha) 
                p = cw * sum((r - self.cbui(user_id, item_id, t))*self.x[item_id] \
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

                    self.x[item_id] += step_size * (cw * \
                            (r - self.cbui(user_id, item_id, t))*s - reg*self.x[item_id])

            logging.info("%s: %s, %s", iter, (rmse / n) ** 0.5, tot)

            if iter >= min_iter and tot > last_tot:
                logging.info("Stopping early")
                return

            last_tot = tot

if __name__ == '__main__':

    logging.basicConfig(level=logging.ERROR)#INFO)

    '''
    train = GroupLensDataSet("ml-100k/u1.base", "\t")    
    model = CFModelBase().load("model.100k-1[0.93205195459].dump")
    #model.baselines(train, 0, 0)

    data = GroupLensDataSet("ml-100k/u.data", "\t")
    u = random.randrange(data.n)
    user_id = train.user_ids[data.rev_user_ids[u]]
    
    plot_x1, plot_y1 = [], []
    plot_x2, plot_y2 = [], []
    plot_x3, plot_y3 = [], []

    last_gap = None
    for item_id, r, timestamp in data.r_u[u]:
        item_id = train.item_ids[data.rev_item_ids[item_id]]
        gap = r - model.rui(user_id, item_id, data.r_u[u]) #model.cbui(user_id, item_id, timestamp)

        if last_gap != None:
            plot_x1.append(timestamp)
            plot_x2.append(timestamp)
            plot_x3.append(timestamp)
            #plot_y1.append((gap-last_gap)*model.wij(last_item_id, item_id))
            #plot_y2.append(gap-last_gap)
            #plot_y3.append(model.wij(last_item_id, item_id))
            plot_y1.append(gap)
            plot_y2.append(gap-last_gap)
            plot_y3.append(model.wij(last_item_id, item_id))

        last_gap = gap
        last_item_id = item_id

    fig = plt.figure()
    p = fig.add_subplot(3, 1, 1)
    p.scatter(plot_x1, plot_y1, c='b', marker='o')
    p = fig.add_subplot(3, 1, 2)
    p.scatter(plot_x2, plot_y2, c='r', marker='s')
    p = fig.add_subplot(3, 1, 3)
    p.scatter(plot_x3, plot_y3, c='r', marker='x')
    fig.show()

    import pdb; pdb.set_trace()
    '''

    model = CFModel()
    
    f = lambda reg, reg_i, reg_u, reg_it, width_mu, width_it: validate(CFModel(), GroupLensDataSet("ml-100k/u1.base", "\t"), GroupLensDataSet("ml-100k/u1.test", "\t"), save=False, reg=reg, reg_i=reg_i, reg_u=reg_u, reg_it=reg_it, width_mu=width_mu, width_it=width_it)

    print full_optimization(f, [(0.0025,), (0,15), (25,), (20,40,70,100,200), (20000,), (250,500,1000)])

    """
    ######
    # 5-fold cross validation of ml-100k
    ######
    
    avg_rmse = 0
    for k in range(1, 6):

        train = GroupLensDataSet("ml-100k/u%s.base"%k, "\t")
        test = GroupLensDataSet("ml-100k/u%s.test"%k, "\t")
        model = CFModel()
        #model.load("model.100k-1[0.92706503318].dump")

        #model.baselines(train, reg_i=25, reg_u=30, reg_it=10, width_mu=5000, width_it=100)
        rmse = validate(model, train, test, save=False, reg=0.0025, reg_i=0, reg_u=25, reg_it=30, width_mu=20000, width_it=20)
        model.save("model.100k-%s[%s].dump" % (k, rmse))

        '''
        for i in range(4):
            x, y1, y2, y3, y4, y5, y6 = [], [], [], [], [], [], []
            for u, r, t in train.r_i[i]:
                x.append(t)
                y1.append(model.mu.query(t))
                y2.append(model.cbu[u])
                y3.append(model.cbi[i])
                y4.append(model.cbit[i].query(t))
                y5.append(r)
                y6.append(model.cbui(u, i, t))
            fig = plt.figure()
            p = fig.add_subplot(5, 1, 1)
            p.scatter(x, y1)
            p = fig.add_subplot(5, 1, 2)
            p.scatter(x, y2)
            p = fig.add_subplot(5, 1, 3)
            p.scatter(x, y3)
            p = fig.add_subplot(5, 1, 4)
            p.scatter(x, y4)
            p = fig.add_subplot(5, 1, 5)
            p.scatter(x, y5, c='g')
            p.scatter(x, y6, c='r')
            fig.show()

        import pdb; pdb.set_trace()
        '''

        avg_rmse += rmse / 5.0
        print k, rmse
   
    print avg_rmse
    """

