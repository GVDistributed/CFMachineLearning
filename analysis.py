#!/usr/bin/python

import matplotlib.pyplot as plt

import sys
import logging
from collections import defaultdict

from baseline import *

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 4:
        print "Usage: %s model-file train-dataset test-dataset" % sys.argv[0]
        sys.exit(-1)

    model = CFModel()
    model.load(sys.argv[1])
    train = GroupLensDataSet(sys.argv[2], "\t")
    test = GroupLensDataSet(sys.argv[3], "\t")

    fig = plt.figure()

    '''
    user_points = defaultdict(int)
    user_x, user_y = [], []
    item_x, item_y = [], []

    for user_id, item_id, r, t in train.iter_ratings():
        rp = model.rui(user_id, item_id, train.r_u[user_id])
        residual = rp - r
        
        user_x.append(user_points[user_id])
        user_y.append(residual)
        user_points[user_id] += 1
        item_x.append(t)
        item_y.append(residual)

    pl1 = fig.add_subplot(2, 2, 1)
    pl1.stem(user_x, user_y)
    pl2 = fig.add_subplot(2, 2, 2)
    pl2.stem(item_x, item_y)
    '''

    user_points = defaultdict(int)
    user_x, user_y = [], []
    item_x, item_y = [], []

    for user_id, item_id, r, t in test.iter_ratings(train):
        rp = model.rui(user_id, item_id, train.r_u[user_id])
        residual = rp - r
        
        user_x.append(user_points[user_id])
        user_y.append(residual)
        user_points[user_id] += 1
        item_x.append(t)
        item_y.append(residual)

    pl1 = fig.add_subplot(2, 2, 3)
    pl1.stem(user_x, user_y)
    pl2 = fig.add_subplot(2, 2, 4)
    pl2.stem(item_x, item_y)

    plt.show()
