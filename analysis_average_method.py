#!/usr/bin/python

import matplotlib.pyplot as plt
from average_method import CFModel, GroupLensDataSet
import random

data = GroupLensDataSet("ml-100k/u.data", "\t")
model = CFModel()
model.baselines(data, reg_i=15, reg_u=25, reg_it=15, width_mu=20000, width_it=20)

##################### 

item_id = 531 # random.randrange(data.m)

x, ratings, user_baseline, item_baseline, b_ui = [], [], [], [], []
for user_id, rating, t in data.r_i[item_id]:
    x.append(t)
    ratings.append(rating)
    user_baseline.append(model.cbu[user_id])
    item_baseline.append(model.cbi[item_id] + model.cbit[item_id].query(t))
    b_ui.append(model.cbui(user_id, item_id, t))

fig = plt.figure()
p = fig.add_subplot(1,1,1)
p.scatter(x, ratings, c='b', marker='o', label="r_ui(t)")
p.scatter(x, b_ui, c='c', marker='o', label="b_ui(t)")
p.scatter(x, user_baseline, c='m', marker='s', label="b_u")
p.scatter(x, item_baseline, c='g', marker='s', label="b_i + b_i(t)")
plt.title("Baseline Decomposition For Item #%s Over Time" % item_id)
plt.ylabel("Ratings")
plt.xlabel("t")
plt.legend(title="Parameters", loc="lower right")
fig.show()

##################### 

x, y = model.mu.to_plot(5000)

fig = plt.figure()
p = fig.add_subplot(1,1,1)
p.plot(*model.mu.to_plot(5000), c='c', linewidth=1, label="5000")
p.plot(*model.mu.to_plot(10000), c='g', linewidth=2, label="10000")
p.plot(*model.mu.to_plot(20000), c='b', linewidth=3, label="20000")
plt.title("Average Ratings Over Time")
plt.ylabel(unichr(956) + "(t)")
plt.xlabel("t")
plt.legend(title='Window Size')
fig.show()

raw_input("")

