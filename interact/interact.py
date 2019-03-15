import os
import sys
import time
import types
import multiprocessing
import cv2
from tqdm import tqdm

class Interact(object):
    EVENT_LBUTTONDOWN = 1
    EVENT_LBUTTONUP = 2
    EVENT_RBUTTONDOWN = 5
    EVENT_RBUTTONUP = 6
    EVENT_MOUSEWHEEL = 10
    
    def __init__(self):
        self.named_windows = {}
        self.capture_mouse_windows = {}
        self.capture_keys_windows = {}        
        self.mouse_events = {}
        self.key_events = {}
        self.pg_bar = None
    
    def log_info(self, msg, end='\n'):
        print (msg, end=end)
        
    def log_err(self, msg, end='\n'):
        print (msg, end=end)        
    
    def named_window(self, wnd_name):
        if wnd_name not in self.named_windows:
            #we will show window only on first show_image
            self.named_windows[wnd_name] = 0
            
        else: print("named_window: ", wnd_name, " already created.")
        
    def destroy_all_windows(self):
        if len( self.named_windows ) != 0:
            cv2.destroyAllWindows()
            self.named_windows = {}
            self.capture_mouse_windows = {}
            self.capture_keys_windows = {}
            self.mouse_events = {}
            self.key_events = {}
        
    def show_image(self, wnd_name, img):
        if wnd_name in self.named_windows:
            if self.named_windows[wnd_name] == 0:
                self.named_windows[wnd_name] = 1
                cv2.namedWindow(wnd_name)                
                if wnd_name in self.capture_mouse_windows:
                    self.capture_mouse(wnd_name)
                
            cv2.imshow (wnd_name, img)
        else: print("show_image: named_window ", wnd_name, " not found.")

    def capture_mouse(self, wnd_name):
        def onMouse(event, x, y, flags, param):
            (inst, wnd_name) = param            
            if event == cv2.EVENT_LBUTTONDOWN: ev = Interact.EVENT_LBUTTONDOWN
            elif event == cv2.EVENT_LBUTTONUP: ev = Interact.EVENT_LBUTTONUP
            elif event == cv2.EVENT_RBUTTONDOWN: ev = Interact.EVENT_RBUTTONDOWN
            elif event == cv2.EVENT_RBUTTONUP: ev = Interact.EVENT_RBUTTONUP
            elif event == cv2.EVENT_MOUSEWHEEL: ev = Interact.EVENT_MOUSEWHEEL
            
            else: ev = 0
            inst.add_mouse_event (wnd_name, x, y, ev, flags) 

        if wnd_name in self.named_windows:
            self.capture_mouse_windows[wnd_name] = True            
            if self.named_windows[wnd_name] == 1:
                cv2.setMouseCallback(wnd_name, onMouse, (self,wnd_name) )
        else: print("capture_mouse: named_window ", wnd_name, " not found.")

    def capture_keys(self, wnd_name):
        if wnd_name in self.named_windows:
            if wnd_name not in self.capture_keys_windows:
                self.capture_keys_windows[wnd_name] = True
            else: print("capture_keys: already set for window ", wnd_name)
        else: print("capture_keys: named_window ", wnd_name, " not found.")

    def progress_bar(self, desc, total, leave=True):
        if self.pg_bar is None:
            self.pg_bar = tqdm( total=total, desc=desc, leave=leave, ascii=True )
        else: print("progress_bar: already set.")

    def progress_bar_inc(self, c):
        if self.pg_bar is not None:
            self.pg_bar.n += c
            self.pg_bar.refresh()
        else: print("progress_bar not set.")

    def progress_bar_close(self):
        if self.pg_bar is not None:
            self.pg_bar.close()
            self.pg_bar = None
        else: print("progress_bar not set.")
    
    def progress_bar_generator(self, data, desc, leave=True):
        for x in tqdm( data, desc=desc, leave=leave, ascii=True ):
            yield x
    
    def process_messages(self, sleep_time=0):
        has_windows = False
        has_capture_keys = False

        if len(self.named_windows) != 0:
            has_windows = True
            
        if len(self.capture_keys_windows) != 0:
            has_capture_keys = True
                
        if has_windows or has_capture_keys:
            wait_key_time = max(1, int(sleep_time*1000) )
            key = cv2.waitKey(wait_key_time) & 0xFF
        else:
            if sleep_time != 0:
                time.sleep(sleep_time)
        
        if has_capture_keys and key != 255:
            for wnd_name in self.capture_keys_windows:
                self.add_key_event (wnd_name, key)
    
    def wait_any_key(self):
        cv2.waitKey(0)
    
    def add_mouse_event(self, wnd_name, x, y, ev, flags):
        if wnd_name not in self.mouse_events: 
            self.mouse_events[wnd_name] = []
        self.mouse_events[wnd_name] += [ (x, y, ev, flags) ]
        
    def add_key_event(self, wnd_name, key):
        if wnd_name not in self.key_events: 
            self.key_events[wnd_name] = []
        self.key_events[wnd_name] += [ (key,) ]

    def get_mouse_events(self, wnd_name):
        ar = self.mouse_events.get(wnd_name, [])
        self.mouse_events[wnd_name] = []        
        return ar
        
    def get_key_events(self, wnd_name):
        ar = self.key_events.get(wnd_name, [])
        self.key_events[wnd_name] = []        
        return ar
        
    def input_number(self, s, default_value, valid_list=None, help_message=None):
        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    raise ValueError("")
                    
                if help_message is not None and inp == '?':
                    print (help_message)
                    continue
                
                i = float(inp)
                if (valid_list is not None) and (i not in valid_list):
                    return default_value
                return i
            except:
                print (default_value)
                return default_value
                
    def input_int(self,s, default_value, valid_list=None, help_message=None):
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
            
    def input_bool(self, s, default_value, help_message=None):
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
            
    def input_str(self, s, default_value, valid_list=None, help_message=None):
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
                
    def input_process(self, stdin_fd, sq, str):
        sys.stdin = os.fdopen(stdin_fd)
        try:
            inp = input (str)
            sq.put (True)
        except:
            sq.put (False)
            
    def input_in_time (self, str, max_time_sec):
        sq = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.input_process, args=( sys.stdin.fileno(), sq, str))
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

interact = Interact()