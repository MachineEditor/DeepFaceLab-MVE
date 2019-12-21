import multiprocessing
import threading
import time

import numpy as np


class IndexHost():  
    """
    Provides random shuffled indexes for multiprocesses
    """
    def __init__(self, indexes_count):                
        self.sq = multiprocessing.Queue()
        self.cqs = []
        self.clis = []        
        self.thread = threading.Thread(target=self.host_thread, args=(indexes_count,) )
        self.thread.daemon = True
        self.thread.start()
        
    def host_thread(self, indexes_count):
        idxs = [*range(indexes_count)]
        shuffle_idxs = []
        sq = self.sq
        
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
                self.cqs[cq_id].put (result)
                    
            time.sleep(0.005)
        
    def create_cli(self):
        cq = multiprocessing.Queue()
        self.cqs.append ( cq ) 
        cq_id = len(self.cqs)-1
        return IndexHost.Cli(self.sq, cq, cq_id)
        
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
            
        def get(self, count):
            self.sq.put ( (self.cq_id,count) )
            
            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)
                
class ListHost():  
    def __init__(self, list_):        
        self.sq = multiprocessing.Queue()
        self.cqs = []
        self.clis = []        
        self.list_ = list_
        self.thread = threading.Thread(target=self.host_thread)
        self.thread.daemon = True
        self.thread.start()
        
    def host_thread(self):
        sq = self.sq
        while True:            
            while not sq.empty():
                obj = sq.get()
                cq_id, cmd = obj[0], obj[1]
                if cmd == 0:
                    item = self.list_[ obj[2] ]
                    self.cqs[cq_id].put ( item )
                    
                elif cmd == 1:
                    self.cqs[cq_id].put ( len(self.list_) )
            time.sleep(0.005)
        
    def create_cli(self):
        cq = multiprocessing.Queue()
        self.cqs.append ( cq ) 
        cq_id = len(self.cqs)-1
        return ListHost.Cli(self.sq, cq, cq_id)
        
    def __len__(self):
        return len(self.list_)
        
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
            
        def __getitem__(self, key):
            self.sq.put ( (self.cq_id,0,key) )
            
            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)
                
        def __len__(self):
            self.sq.put ( (self.cq_id,1) )
            
            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)      
                    
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
