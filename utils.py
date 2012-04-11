#!/usr/bin/python

def binary_search(x, val, key=lambda i: i):
    low, high = 0, len(x) - 1
    while low < high - 1:
        mid = (low + high) / 2
        cur = key(x[mid])
        if cur < val:
            low = mid # keep the mid
        elif cur > val:
            high = mid # keep the mid
        else:
            return mid
    # return the closer one
    return min(low, min(low + 1, len(x) - 1), 
        key=lambda i: abs(key(x[i]) - val))

class WindowedAverage(object):

    def __init__(self, size, reg = 0, x = None):
        self.x = x or []
        self.size = size
        self.reg = reg

    def add(self, k, v):
        self.x.append((k, v))

    def process(self):
        self.x.sort()
        s = 0
        for i in range(len(self.x)):
            s += self.x[i][1]
            self.x[i] = (self.x[i][0], s)
        return self

    def csum(self, left, right):
        if left == 0:
            return self.x[right][1]
        return self.x[right][1] - self.x[left - 1][1]

    def avg(self, i, size = None):
        size = size or self.size
        lb, ub = 0, len(self.x) - 1
        if i - size/2 < lb:
            left = lb
            right = min(ub, lb + size)
        elif i + size/2 > ub:
            left = max(ub - size, lb)
            right = ub
        else:
            left = i - size/2
            right = i + size/2
        return self.csum(left, right) / float(right - left + 1 + self.reg)

    def query(self, k, size = None):
        i = binary_search(self.x, k, key=lambda i: i[0])
        return self.avg(i, size)

    def to_plot(self, size = None):
        x, y = [], []        
        for i in range(len(self.x)):
            x.append(self.x[i][0])
            y.append(self.avg(i, size))
        return x, y

if __name__ == '__main__':

    x = range(10, 31, 3)
    for i in range(9, 32):
        print i, x[binary_search(x, i)]
