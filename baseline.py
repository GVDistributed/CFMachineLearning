#!/usr/bin/python

from collections import defaultdict

class CFModel(object):
    def __init__(self):
        self.mu = None
        self.bu = {}
        self.bi = {}
        self.cbu = {}
        self.cbi = {}
        self.q = {}
        self.x = {}
        self.y = {}

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def cbui(self, user_id, item_id):
        return self.mu + self.cbu[user_id] + self.cbi[item_id]

    def bui(self, user_id, item_id):
        return self.mu + self.bu[user_id] + self.bi[item_id]
  

class GroupLensTrainingSet(object):

    def __init__(self, filename):
        self.r_u = defaultdict(list)
        self.r_i = defaultdict(list)
        with open(filename) as file_read:
            for line in file_read:
                user_id, item_id, r, timestamp = line.split('::')
                user_id = int(user_id)
                item_id = int(item_id)
                r = float(r) # 0.5 increments
                timestamp = int(timestamp)
                self.r_u[user_id].append((item_id, r, timestamp))
                self.r_i[item_id].append((user_id, r, timestamp))
        
        for rating_list in itertools.chain(self.r_u.itervalues(), self.r_i.itervalues()):
            rating_list.sort(key = lambda x: x[2])

    def iter_users(self):
        return self.r_u.iterkeys()

    def iter_items(self):
        return self.r_i.iterkeys()

    def iter_ratings(self):
        for user_id, ratings in self.r_u.iteritems():
            for item_id, r, timestamp in ratings:
                yield user_id, item_id, r, timestamp

def cf_train(data, reg_i, reg_u):
    model = CFModel()
     
    # Compute Mu    
    t = 0
    n = 0

    for item_id, ratings in data.iter_items():
        for user_id, r, timestamp in ratings:
            
    for user_id in data.iter_users():
        m

if __name__ == '__main__':

    data = GroupLensTrainingSet("../../ml-10M100K/r1.train")
    cf_train(data, 25, 10)

