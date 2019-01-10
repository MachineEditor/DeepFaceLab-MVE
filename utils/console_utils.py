import os
import sys
import time
import multiprocessing

def input_int(s, default_value, valid_list=None, help_message=None):
    while True:
        try:
            inp = input(s)
            if len(inp) == 0:
                raise ValueError("")
                
            if help_message is not None and inp == '?':
                print (help_message)
                continue
            
            i = int(inp)
            if (valid_list is not None) and (i not in valid_list):
                return default_value
            return i
        except:
            print (default_value)
            return default_value
        
def input_bool(s, default_value, help_message=None):
    while True:
        try:
            inp = input(s)
            if len(inp) == 0:
                raise ValueError("")
                
            if help_message is not None and inp == '?':
                print (help_message)
                continue
                
            return bool ( {"y":True,"n":False,"1":True,"0":False}.get(inp.lower(), default_value) )
        except:
            print ( "y" if default_value else "n" )
            return default_value
        
def input_str(s, default_value, valid_list=None, help_message=None):
    while True:        
        try:
            inp = input(s)
            if len(inp) == 0:
                raise ValueError("")
                
            if help_message is not None and inp == '?':
                print (help_message)
                continue
                
            if (valid_list is not None) and (inp.lower() not in valid_list):
                return default_value
            return inp
        except:
            print (default_value)
            return default_value
            
def input_process(stdin_fd, sq, str):
    sys.stdin = os.fdopen(stdin_fd)
    try:
        inp = input (str)
        sq.put (True)
    except:
        sq.put (False)
        
def input_in_time (str, max_time_sec):
    sq = multiprocessing.Queue()
    p = multiprocessing.Process(target=input_process, args=( sys.stdin.fileno(), sq, str))
    p.start()
    t = time.time()
    inp = False
    while True:
        if not sq.empty():
            inp = sq.get()
            break
        if time.time() - t > max_time_sec:
            break
    p.terminate()
    sys.stdin = os.fdopen( sys.stdin.fileno() )
    return inp