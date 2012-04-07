#!/usr/bin/python

import sys
import cPickle
import logging
import random
import itertools
from collections import defaultdict

from numpy import *

class GroupLensDataSet(object):

    def __init__(self, filename, delim="::"):
        logging.info("Loading dataset")

        self.user_ids = {}
        self.item_ids = {}
        self.rev_user_ids = []
        self.rev_item_ids = []
        self.r_u = []
        self.r_i = []
        with open(filename) as file_read:
            for i, line in enumerate(file_read):
                user_id, item_id, r, timestamp = line.split(delim)
                user_id = int(user_id)
                item_id = int(item_id)
                r = float(r) # 0.5 increments
                timestamp = int(timestamp)

                if not self.user_ids.has_key(user_id):
                    self.r_u.append([])
                    self.user_ids[user_id] = len(self.user_ids)
                    self.rev_user_ids.append(user_id)
                uid = self.user_ids[user_id]

                if not self.item_ids.has_key(item_id):
                    self.r_i.append([])
                    self.item_ids[item_id] = len(self.item_ids)
                    self.rev_item_ids.append(item_id)
                iid = self.item_ids[item_id]

                self.r_u[uid].append((iid, r, timestamp))
                self.r_i[iid].append((uid, r, timestamp))
        
                if (i + 1) % 10000 == 0:
                    logging.info("Read %d", i + 1)

        for rating_list in itertools.chain(self.r_u, self.r_i):
            rating_list.sort(key = lambda x: x[2])
        
        self.n = len(self.r_u)
        self.m = len(self.r_i)

        logging.info("%d users and %d items", self.n, self.m)

    def indexes_to_ids(self, u, i):
        return (self.rev_user_ids[u], self.rev_item_ids[i])

    def ids_to_indexes(self, user_id, item_id):
        return (self.user_ids[user_id], self.item_ids[item_id])

    def iter_users(self):
        return enumerate(self.r_u)

    def iter_items(self):
        return enumerate(self.r_i)

    def iter_ratings(self, baseset=None):
        for user_id, ratings in enumerate(self.r_u):
            for item_id, r, timestamp in ratings:
                if baseset:
                    # Convert ids from one dataset to the other
                    try:
                        u, i = baseset.ids_to_indexes(*self.indexes_to_ids(user_id, item_id))
                    except KeyError:
                        logging.warn("User %s or Item %s missing from dataset", user_id, item_id)
                        continue
                else:
                    u, i = user_id, item_id
                yield u, i, r, timestamp

class CFModel(object):
    def __init__(self, alpha=0.5, f=200):
        self.mu = 0
        self.bu = array([])
        self.bi = array([])
        self.cbu = array([])
        self.cbi = array([])
        self.q = None
        self.x = None

        self.alpha = alpha
        self.f = f

    def load(self, filename):
        with open(filename) as file_read:
            self.mu, self.bu, self.bi, self.cbu, self.cbi, self.q, self.x, self.alpha, self.f = cPickle.loads(file_read.read())

    def save(self, filename):
        with open(filename, "w") as file_write:
            file_write.write(cPickle.dumps((self.mu, self.bu, self.bi, self.cbu, self.cbi, self.q, self.x, self.alpha, self.f), protocol=-1))

    def cbui(self, user_id, item_id):
        return self.mu + self.cbu[user_id] + self.cbi[item_id]

    def bui(self, user_id, item_id):
        return self.mu + self.bu[user_id] + self.bi[item_id]

    def wij(self, i, j):
        return dot(self.q[i], self.x[j])

    def rui(self, user_id, item_id, ratings):
        p = len(ratings)**(-self.alpha) * \
            sum((r - self.cbui(user_id, item_id))*self.x[item_id] for item_id, r, t in ratings)

        return self.bui(user_id, item_id) + dot(self.q[item_id], p)

    def train(self, data, reg, reg_i, reg_u, min_iter, max_iter, step_size):
        logging.info("Computing mu...")  
        t = 0
        n = 0
        for v1, v2, r, v3 in data.iter_ratings():
            t += r
            n += 1
        self.mu = t / float(n)
        logging.info("%s", self.mu)

        logging.info("Computing item baselines...")
        self.cbi = []
        for ratings in data.r_i:
            t = 0
            n = 0
            for user_id, r, timestamp in ratings:
                t += (r - self.mu)
                n += 1
            self.cbi.append(t / float(reg_i + n))
        self.cbi = array(self.cbi)

        logging.info("Computing user baselines...")
        self.cbu = []
        for user_id, ratings in data.iter_users():
            t = 0
            n = 0
            for item_id, r, timestamp in ratings:
                t += (r - self.cbi[item_id] - self.mu)
                n += 1
            self.cbu.append(t / float(reg_u + n))
        self.cbu = array(self.cbu)

        logging.info("Performing optimization...")
        self.bi = array(self.cbi)
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
                p = len(data.r_u[user_id])**(-self.alpha) * \
                    sum((r - self.cbui(user_id, item_id))*self.x[item_id] \
                        for item_id, r, t in data.r_u[user_id])

                s = 0
                for item_id, r, t in data.r_u[user_id]:
                    rp = self.bui(user_id, item_id) + dot(self.q[item_id], p)
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
                            (r - self.cbui(user_id, item_id))*s - reg*self.x[item_id])

            logging.info("%s: %s, %s", iter, (rmse / n) ** 0.5, tot)

            if iter >= min_iter and tot > last_tot:
                logging.info("Stopping early")
                return

            last_tot = tot

def silly_optimization(f, args):
    best_args = [(arg[0]+arg[1])/2.0 for arg in args]
    best_val = float('inf')

    for i, (low, high, step) in enumerate(args):
       cur_args = best_args[:]
       while low <= high:
           cur_args[i] = low
           try:
               val = f(*cur_args) 
           except Exception as e:
               logging.error("Bad call: %s", cur_args)
               logging.exception(e)
               val = float('inf')

           logging.info("%s=%s", cur_args, val)
           if val < best_val:
               best_args = cur_args[:]
               best_val = val

           low += step

    return best_args, best_val

def full_optimization(f, arglists):
    best_args = None
    best_val = float('inf')
    for cur_args in itertools.product(*arglists):
       try:
           val = f(*cur_args) 
       except Exception as e:
           logging.error("Bad call: %s", cur_args)
           logging.exception(e)
           val = float('inf')

       logging.info("%s=%s", cur_args, val)
       if val < best_val:
           best_args = cur_args[:]
           best_val = val

    return best_args, best_val

def validate(model, train, test, reg, reg_i, reg_u, 
             min_iter=10, max_iter=100, step_size=0.01, save=True):

    model.train(train, reg, reg_i, reg_u, min_iter, max_iter, step_size)

    tot = 0
    n = 0
    for user_id, item_id, r, t in test.iter_ratings(train):
        rp = model.rui(user_id, item_id, train.r_u[user_id])
        tot += (r - rp)**2
        n += 1
    rmse = (tot / float(n)) ** 0.5

    if save:
        model.save("model[%s-%s-%s-%s].dump" % (rmse, reg, reg_i, reg_u))

    return rmse

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ######
    # Example for loading a model
    ######

    '''
    model = CFModel()
    model.load("model.backup")
    '''

    ######
    # Example for loading datasets
    ######

    '''
    train = GroupLensDataSet("ml-100k/u2.base", "\t")
    test = GroupLensDataSet("ml-100k/u2.test", "\t")
    '''

    ######
    # Approach 1 for figuring out the best magic constants
    ######

    '''
    f = lambda *args: validate(model, train, test, *args)
    closed_f = (lambda model, train, test: f)(model, train, test)
    args = [(0.01, 2.00, 0.02), (0, 40, 1), (0, 40, 1)]
    print silly_optimization(closed_f, args)
    '''

    ######
    # Approach 2 for figuring out the best magic constants
    ######

    '''
    f = lambda *args: validate(CFModel(), train, test, *args)
    closed_f = (lambda train, test: f)(train, test)
    arglists = [[0.002, 0.0025, 0.003], [10, 15, 20], [20, 25, 30]]
    print full_optimization(closed_f, arglists)
    '''

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
    
