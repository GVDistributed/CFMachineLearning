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

class Polynomial:
    def __init__(self, deg=1):
        self.deg = deg
        self.coeff = array([(random.random()-0.5) for x in xrange(self.deg)])

    def __str__(self):
        return "Degree %d: %s"%(self.deg,','.join(map(str,self.coeff)))


    def eval(self, t):
        return dot(array([ t**x for x in xrange(self.deg)]), self.coeff)

    def norm(self):
        return dot(self.coeff,self.coeff)

    def update(self, t, step_size, e, reg):
        count = 1
        for x in xrange(self.deg):
            if (x==0):
                self.coeff[x] += step_size*(e - reg*self.coeff[x])
            else:
                self.coeff[x] += step_size*(e - reg*self.coeff[x])*count*x
            count*=t

class CFModel(CFModelBase):

    def bui(self, user_id, item_id, t):
        return self.mu + self.bu[user_id] + self.bi[item_id].eval(t)

    def rui(self, user_id, item_id, time, ratings):
        p = len(ratings)**(-self.alpha) * \
            sum((r - self.cbui(user_id, item_id))*self.x[item_id] for item_id, r, t in ratings)

        return self.bui(user_id, item_id, time) + dot(self.q[item_id], p)

    def train(self, data, reg, reg_i, reg_u, min_iter, max_iter, step_size, poly_deg):
        self.mu = self.compute_mu(data)
        logging.info("%s", self.mu)

        self.cbi = self.compute_bi(data, reg_i)
        self.cbu = self.compute_bu(data, reg_u)

        logging.info("Performing Stochastic Gradient Descent...")
        self.bi = [Polynomial(poly_deg) for x in xrange(len(data.r_i))]
        self.bu = array(self.cbu)
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
                confidence_weight = len(data.r_u[user_id])**(-self.alpha) 
                p = confidence_weight * \
                    sum((r - self.cbui(user_id, item_id))*self.x[item_id] \
                        for item_id, r, t in data.r_u[user_id])

                s = 0
                for item_id, r, t in data.r_u[user_id]:
                    rp = self.bui(user_id, item_id, t) + dot(self.q[item_id], p)
                    e = r - rp

                    tot += e**2 + reg*(self.bu[user_id]**2 + self.bi[item_id].norm() + \
                        dot(self.q[item_id], self.q[item_id]))

                    s += e*self.q[item_id]
                    self.q[item_id] += step_size * (e*p - reg*self.q[item_id])
                    self.bu[user_id] += step_size * (e - reg*self.bu[user_id])
                    self.bi[item_id].update(t, step_size , e , reg)

                    rmse += e**2
                    n += 1
 
                for item_id, r, t in data.r_u[user_id]:
                    tot += reg * len(data.r_u[user_id]) * \
                        dot(self.x[item_id], self.x[item_id])

                    self.x[item_id] += step_size * (confidence_weight * \
                            (r - self.cbui(user_id, item_id))*s - reg*self.x[item_id])

            logging.info("%s: %s, %s", iter, (rmse / n) ** 0.5, tot)

            if iter >= min_iter and tot > last_tot:
                logging.info("Stopping early")
                logging.info("A couple of random b_i")
                for x in xrange(8):
                    random_item_id = int(random.random()*len(self.bi))
                    logging.info(self.bi[random_item_id])

                return

            last_tot = tot

def validate(model, train, test, reg, reg_i, reg_u, 
             min_iter=10, max_iter=60, step_size=0.005, save=True):

    model.train(train, reg, reg_i, reg_u, min_iter, max_iter, step_size, 3)

    tot = 0
    n = 0
    for user_id, item_id, r, t in test.iter_ratings(train):
        rp = model.rui(user_id, item_id, t, train.r_u[user_id])
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
        avg_rmse += rmse / 5.0
        print k, rmse
    print avg_rmse
