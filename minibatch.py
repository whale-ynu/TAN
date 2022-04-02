import numpy as np
from queue import Queue

class MinibatchIterator(object):

    def __init__(self, file,batch_size):
        f=open(file,'r+')
        datas=f.readlines()
        f.close()
        i=0
        self.data=[[] for i in range(len(datas))]
        for line in datas:
            tmp=line.strip().split(' ')
            self.data[i].append(tmp[0])
            self.data[i].append(tmp[1])
            self.data[i].append(tmp[2])
            self.data[i].append(tmp[3])
            self.data[i].append(tmp[4])
            i=i+1
        self.data=np.array(self.data)
        self.batch_size = batch_size
        self.result_queue = Queue(10000000)

    def batch_num(self):
        batchs = len(self.data[:,0]) / self.batch_size
        tmp = len(self.data[:,0]) % self.batch_size
        if tmp == 0:
            total_batchs = int(batchs)
        else:
            total_batchs = int(batchs) + 1
        return total_batchs

    def shuffle(self, a, b, c,d,e):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)
    def generate_batch(self):
        self.shuffle(self.data[:, 0], self.data[:, 1], self.data[:, 2], self.data[:, 3], self.data[:, 4])
        self.batchs = self.batch_num()
        for i in range(self.batchs):
            self.result_queue.put(i)
    def is_empty(self):
        return self.result_queue.empty()
    def next_batch(self):
        i = self.result_queue.get()
        X_U = []
        X_UAS = []
        X_S = []
        X_SAS = []
        Y = []
        if(i==self.batchs-1):
            start = (i-1) * self.batch_size
        else:
            start = i * self.batch_size
        while len(X_U) < self.batch_size and start < len(self.data[:, 0]):
            #if float(self.data[start, 4])<=350:
            Y.append(float(self.data[start, 4]))
            X_U.append(self.data[start, 0])
            X_UAS.append(self.data[start, 1])
            X_S.append(self.data[start, 2])
            X_SAS.append(self.data[start, 3])
            start=start+1
            
        return X_U, X_UAS, X_S, X_SAS, Y




