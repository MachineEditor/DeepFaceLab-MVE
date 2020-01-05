import multiprocessing
import threading
import time
import traceback
import numpy as np


class Index2DHost():
    """
    Provides random shuffled 2D indexes for multiprocesses
    """
    def __init__(self, indexes2D, max_number_of_clis=128):
        self.sq = multiprocessing.Queue()
        self.cqs = [ multiprocessing.Queue() for _ in range(max_number_of_clis) ]
        self.n_clis = 0
        self.max_number_of_clis = max_number_of_clis

        self.p = multiprocessing.Process(target=self.host_proc, args=(indexes2D, self.sq, self.cqs)  )
        self.p.daemon = True
        self.p.start()
        
    def host_proc(self, indexes2D, sq, cqs):
        indexes_counts_len = len(indexes2D)

        idxs = [*range(indexes_counts_len)]
        idxs_2D = [None]*indexes_counts_len
        shuffle_idxs = []
        shuffle_idxs_2D = [None]*indexes_counts_len
        for i in range(indexes_counts_len):
            idxs_2D[i] = indexes2D[i]
            shuffle_idxs_2D[i] = []

        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, cmd = obj[0], obj[1]

                if cmd == 0: #get_1D
                    count = obj[2]

                    result = []
                    for i in range(count):
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = idxs.copy()
                            np.random.shuffle(shuffle_idxs)
                        result.append(shuffle_idxs.pop())
                    cqs[cq_id].put (result)
                elif cmd == 1: #get_2D
                    targ_idxs,count = obj[2], obj[3]
                    result = []

                    for targ_idx in targ_idxs:
                        sub_idxs = []
                        for i in range(count):
                            ar = shuffle_idxs_2D[targ_idx]
                            if len(ar) == 0:
                                ar = shuffle_idxs_2D[targ_idx] = idxs_2D[targ_idx].copy()
                                np.random.shuffle(ar)
                            sub_idxs.append(ar.pop())
                        result.append (sub_idxs)
                    cqs[cq_id].put (result)

            time.sleep(0.005)

    def create_cli(self):
        cq = multiprocessing.Queue()
        self.cqs.append ( cq )
        cq_id = len(self.cqs)-1
        return Index2DHost.Cli(self.sq, cq, cq_id)

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

    class Cli():
        def __init__(self, sq, cq, cq_id):
            self.sq = sq
            self.cq = cq
            self.cq_id = cq_id

        def get_1D(self, count):
            self.sq.put ( (self.cq_id,0, count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

        def get_2D(self, idxs, count):
            self.sq.put ( (self.cq_id,1,idxs,count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

class IndexHost():
    """
    Provides random shuffled indexes for multiprocesses
    """
    def __init__(self, indexes_count, max_number_of_clis=128):
        self.sq = multiprocessing.Queue()
        self.cqs = [ multiprocessing.Queue() for _ in range(max_number_of_clis) ]
        self.n_clis = 0
        self.max_number_of_clis = max_number_of_clis

        self.p = multiprocessing.Process(target=self.host_proc, args=(indexes_count, self.sq, self.cqs)  )
        self.p.daemon = True
        self.p.start()

    def host_proc(self, indexes_count, sq, cqs):
        idxs = [*range(indexes_count)]
        shuffle_idxs = []

        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, count = obj[0], obj[1]

                result = []
                for i in range(count):
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = idxs.copy()
                        np.random.shuffle(shuffle_idxs)
                    result.append(shuffle_idxs.pop())
                cqs[cq_id].put (result)

            time.sleep(0.005)

    def create_cli(self):
        if self.n_clis == self.max_number_of_clis:
            raise Exception("")
        
        cq_id = self.n_clis
        self.n_clis += 1    
        
        return IndexHost.Cli(self.sq, self.cqs[cq_id], cq_id)

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

    class Cli():
        def __init__(self, sq, cq, cq_id):
            self.sq = sq
            self.cq = cq
            self.cq_id = cq_id

        def multi_get(self, count):
            self.sq.put ( (self.cq_id,count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

class ListHost():
    def __init__(self, list_=None, max_number_of_clis=128):
        self.sq = multiprocessing.Queue()
        self.cqs = [ multiprocessing.Queue() for _ in range(max_number_of_clis) ]
        self.n_clis = 0
        self.max_number_of_clis = max_number_of_clis
        
        self.p = multiprocessing.Process(target=self.host_proc, args=(self.sq, self.cqs)  )
        self.p.daemon = True
        self.p.start()

    def host_proc(self, sq, cqs):
        m_list = list()
        
        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, cmd = obj[0], obj[1]
                if cmd == 0:
                    cqs[cq_id].put ( len(m_list) )
                elif cmd == 1:
                    idx = obj[2]                    
                    item = m_list[idx ]
                    cqs[cq_id].put ( item )
                elif cmd == 2:
                    result = []
                    for item in obj[2]:
                        result.append ( m_list[item] )
                    cqs[cq_id].put ( result )
                elif cmd == 3:
                    m_list.insert(obj[2], obj[3])
                elif cmd == 4:
                    m_list.append(obj[2])
                elif cmd == 5:
                    m_list.extend(obj[2])    
            time.sleep(0.005)

    def create_cli(self):       
        if self.n_clis == self.max_number_of_clis:
            raise Exception("")
        
        cq_id = self.n_clis
        self.n_clis += 1    
        
        return ListHost.Cli(self.sq, self.cqs[cq_id], cq_id)

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

    class Cli():
        def __init__(self, sq, cq, cq_id):
            self.sq = sq
            self.cq = cq
            self.cq_id = cq_id

        def __len__(self):
            self.sq.put ( (self.cq_id,0) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)
                
        def __getitem__(self, key):
            self.sq.put ( (self.cq_id,1,key) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)
                
        def multi_get(self, keys):
            self.sq.put ( (self.cq_id,2,keys) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)
    
        def insert(self, index, item):
            self.sq.put ( (self.cq_id,3,index,item) )
        
        def append(self, item):
            self.sq.put ( (self.cq_id,4,item) )
            
        def extend(self, items):
            self.sq.put ( (self.cq_id,5,items) )
            
        

class DictHost():
    def __init__(self, d, num_users):
        self.sqs = [ multiprocessing.Queue() for _ in range(num_users) ]
        self.cqs = [ multiprocessing.Queue() for _ in range(num_users) ]

        self.thread = threading.Thread(target=self.host_thread, args=(d,) )
        self.thread.daemon = True
        self.thread.start()

        self.clis = [ DictHostCli(sq,cq) for sq, cq in zip(self.sqs, self.cqs) ]

    def host_thread(self, d):
        while True:
            for sq, cq in zip(self.sqs, self.cqs):
                if not sq.empty():
                    obj = sq.get()
                    cmd = obj[0]
                    if cmd == 0:
                        cq.put (d[ obj[1] ])
                    elif cmd == 1:
                        cq.put ( list(d.keys()) )

            time.sleep(0.005)


    def get_cli(self, n_user):
        return self.clis[n_user]

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

class DictHostCli():
    def __init__(self, sq, cq):
        self.sq = sq
        self.cq = cq

    def __getitem__(self, key):
        self.sq.put ( (0,key) )

        while True:
            if not self.cq.empty():
                return self.cq.get()
            time.sleep(0.001)

    def keys(self):
        self.sq.put ( (1,) )
        while True:
            if not self.cq.empty():
                return self.cq.get()
            time.sleep(0.001)
