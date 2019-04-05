import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import numpy.linalg as npl

import imagelib
from facelib import LandmarksProcessor
from imagelib import IEPolys
from interact import interact as io
from utils import Path_utils
from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG

class MaskEditor:
    STATE_NONE=0
    STATE_MASKING=1
    
    def __init__(self, img, mask=None, ie_polys=None, get_status_lines_func=None):
        self.img = imagelib.normalize_channels (img,3)
        h, w, c = img.shape
        ph, pw = h // 4, w // 4
        
        if mask is not None:
            self.mask = imagelib.normalize_channels (mask,3)
        else:
            self.mask = np.zeros ( (h,w,3) )
        self.get_status_lines_func = get_status_lines_func
        
        self.state_prop = self.STATE_NONE
        
        self.w, self.h = w, h
        self.pw, self.ph = pw, ph
        self.pwh = np.array([self.pw, self.ph])
        self.pwh2 = np.array([self.pw*2, self.ph*2])
        self.sw, self.sh = w+pw*2, h+ph*2

        if ie_polys is None:
            ie_polys = IEPolys()
        self.ie_polys = ie_polys
        
        self.polys_mask = None
        
        self.mouse_x = self.mouse_y = 9999
        self.screen_status_block = None
        self.screen_status_block_dirty = True
        self.screen_changed = True
        
    def set_state(self, state):
        self.state = state
    
    @property
    def state(self):
        return self.state_prop

    @state.setter
    def state(self, value):
        self.state_prop = value
        if value == self.STATE_MASKING:
            self.ie_polys.dirty = True
                  
    def get_mask(self):        
        if self.ie_polys.switch_dirty():
            self.screen_status_block_dirty = True
            self.ie_mask = img = self.mask.copy()

            self.ie_polys.overlay_mask(img)     
                          
            return img
        return self.ie_mask
        
    def get_screen_overlay(self):
        img = np.zeros ( (self.sh, self.sw, 3) )
        
        if self.state == self.STATE_MASKING:
            mouse_xy = self.mouse_xy.copy() + self.pwh

            l = self.ie_polys.n_list()
            if l.n > 0:            
                p = l.cur_point().copy() + self.pwh
                color = (0,1,0) if l.type == 1 else (0,0,1)
                cv2.line(img, tuple(p), tuple(mouse_xy), color )

        return img
    
    def undo_to_begin_point(self):
        while not self.undo_point():
            pass
        
    def undo_point(self):
        self.screen_changed = True
        if self.state == self.STATE_NONE:
            if self.ie_polys.n > 0:                   
                self.state = self.STATE_MASKING
                
        if self.state == self.STATE_MASKING:
            if self.ie_polys.n_list().n_dec() == 0 and \
               self.ie_polys.n_dec() == 0:
                self.state = self.STATE_NONE
            else:
                return False
                    
        return True
        
    def redo_to_end_point(self):
        while not self.redo_point():
            pass
                
    def redo_point(self):
        self.screen_changed = True
        if self.state == self.STATE_NONE:
            if self.ie_polys.n_max > 0:                    
                self.state = self.STATE_MASKING
                if self.ie_polys.n == 0:
                    self.ie_polys.n_inc()
                    
        if self.state == self.STATE_MASKING:
            while True:
                l = self.ie_polys.n_list()
                if l.n_inc() == l.n_max:
                    if self.ie_polys.n == self.ie_polys.n_max:
                        break
                    self.ie_polys.n_inc()
                else:
                    return False
                    
        return True
        
    def combine_screens(self, screens):
    
        screens_len = len(screens)
        
        new_screens = []
        for screen, padded_overlay in screens:
            screen_img = np.zeros( (self.sh, self.sw, 3), dtype=np.float32 )
            
            screen = imagelib.normalize_channels (screen, 3)
            h,w,c = screen.shape
            
            screen_img[self.ph:-self.ph, self.pw:-self.pw, :] = screen
            
            if padded_overlay is not None:
                screen_img = screen_img + padded_overlay
            
            screen_img = np.clip(screen_img*255, 0, 255).astype(np.uint8)
            new_screens.append(screen_img)
            
        return np.concatenate (new_screens, axis=1)
    
    def get_screen_status_block(self, w, c):
        if self.screen_status_block_dirty:
            self.screen_status_block_dirty = False
            lines = [
                    'Polys current/max = %d/%d' % (self.ie_polys.n, self.ie_polys.n_max),
                    ]
            if self.get_status_lines_func is not None:
                lines += self.get_status_lines_func()
                
            lines_count = len(lines)
    
            
            h_line = 21
            h = lines_count * h_line
            img = np.ones ( (h,w,c) ) * 0.1
            
            for i in range(lines_count):
                img[ i*h_line:(i+1)*h_line, 0:w] += \
                    imagelib.get_text_image (  (h_line,w,c), lines[i], color=[0.8]*c )
                
            self.screen_status_block = np.clip(img*255, 0, 255).astype(np.uint8)
            
        return self.screen_status_block
        
    def set_screen_status_block_dirty(self):
        self.screen_status_block_dirty = True
        
    def switch_screen_changed(self):
        result = self.screen_changed
        self.screen_changed = False
        return result
        
    def make_screen(self):
        screen_overlay = self.get_screen_overlay()
        final_mask = self.get_mask()
  
        masked_img = self.img*final_mask*0.5 + self.img*(1-final_mask)

        pink = np.full ( (self.h, self.w, 3), (1,0,1) )        
        pink_masked_img = self.img*final_mask + pink*(1-final_mask)
        
        screens = [ (self.img, screen_overlay),
                    (masked_img, screen_overlay),
                    (pink_masked_img, screen_overlay),                    
                    ]                
        screens = self.combine_screens(screens)
        
        status_img = self.get_screen_status_block( screens.shape[1], screens.shape[2] )
           
        result = np.concatenate ( [screens, status_img], axis=0  )
        
        return result
        
    def mask_finish(self, n_clip=True):
        if self.state == self.STATE_MASKING:
            self.screen_changed = True
            if self.ie_polys.n_list().n <= 2:
                self.ie_polys.n_dec()
            self.state = self.STATE_NONE
            if n_clip:
                self.ie_polys.n_clip()
            
    def set_mouse_pos(self,x,y):
        mouse_x = x % (self.sw) - self.pw
        mouse_y = y % (self.sh) - self.ph
        if mouse_x != self.mouse_x or mouse_y != self.mouse_y:
            self.mouse_xy = np.array( [mouse_x, mouse_y] )
            self.mouse_x, self.mouse_y = self.mouse_xy
            self.screen_changed = True

    def mask_point(self, type):
        self.screen_changed = True
        if self.state == self.STATE_MASKING and \
           self.ie_polys.n_list().type != type:
            self.mask_finish()
            
        elif self.state == self.STATE_NONE:
            self.state = self.STATE_MASKING            
            self.ie_polys.add(type)
        
        if self.state == self.STATE_MASKING:
            self.ie_polys.n_list().add (self.mouse_x, self.mouse_y)            

    def get_ie_polys(self):
        return self.ie_polys
        
def mask_editor_main(input_dir, confirmed_dir=None, skipped_dir=None):
    input_path = Path(input_dir)
    
    confirmed_path = Path(confirmed_dir)
    skipped_path = Path(skipped_dir)

    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    if not confirmed_path.exists():
        confirmed_path.mkdir(parents=True)

    if not skipped_path.exists():
        skipped_path.mkdir(parents=True)

    wnd_name = "MaskEditor tool"
    io.named_window (wnd_name)
    io.capture_mouse(wnd_name)
    io.capture_keys(wnd_name)

    image_paths = [ Path(x) for x in Path_utils.get_image_paths(input_path)]
    done_paths = []
    
    image_paths_total = len(image_paths)

    do_prev_count = 0
    do_save_move_count = 0
    do_save_count = 0
    do_skip_move_count = 0
    do_skip_count = 0
    
    is_exit = False
    while not is_exit:
        
        if len(image_paths) > 0:
            filepath = image_paths.pop(0)
        else:
            filepath = None

        if filepath is not None:
            if filepath.suffix == '.png':
                dflimg = DFLPNG.load( str(filepath) )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load ( str(filepath) )
            else:
                dflimg = None
                
            if dflimg is None:
                io.log_err ("%s is not a dfl image file" % (filepath.name) )
                continue
                
            lmrks = dflimg.get_landmarks()
            ie_polys = dflimg.get_ie_polys()
            
            img = cv2_imread(str(filepath)) / 255.0
            mask = LandmarksProcessor.get_image_hull_mask( img.shape, lmrks)
        else:
            img = np.zeros ( (256,256,3) )
            mask = np.ones ( (256,256,3) )
            ie_polys = None
            
        def get_status_lines_func():
            return ['Progress: %d / %d . Current file: %s' % (len(done_paths), image_paths_total, str(filepath.name) if filepath is not None else "end" ), 
                    '[Left mouse button] - mark include mask.',
                    '[Right mouse button] - mark exclude mask.',
                    '[Middle mouse button] - finish current poly.',
                    '[Mouse wheel] - undo/redo poly or point. [+ctrl] - undo to begin/redo to end',
                    '[q] - prev image. [w] - skip and move to %s. [e] - save and move to %s. ' % (skipped_path.name, confirmed_path.name),
                    '[z] - prev image. [x] - skip. [c] - save. ',
                    'hold [shift] - speed up the frame counter by 10.'
                    '[esc] - quit'
                    ]
        ed = MaskEditor(img, mask, ie_polys, get_status_lines_func)

        next = False
        while not next:
            io.process_messages(0.005)
            
            if do_prev_count + do_save_move_count + do_save_count + do_skip_move_count + do_skip_count == 0:
                for (x,y,ev,flags) in io.get_mouse_events(wnd_name):
                    ed.set_mouse_pos(x, y) 
                    if filepath is not None:
                        if ev == io.EVENT_LBUTTONDOWN:
                            ed.mask_point(1)
                        elif ev == io.EVENT_RBUTTONDOWN:
                            ed.mask_point(0)
                        elif ev == io.EVENT_MBUTTONDOWN:
                            ed.mask_finish()
                        elif ev == io.EVENT_MOUSEWHEEL:
                            if flags & 0x80000000 != 0:
                                if flags & 0x8 != 0:
                                    ed.undo_to_begin_point()
                                else:
                                    ed.undo_point()
                            else:
                                if flags & 0x8 != 0:
                                    ed.redo_to_end_point()
                                else:
                                    ed.redo_point()

                for key, chr_key, ctrl_pressed, alt_pressed, shift_pressed in io.get_key_events(wnd_name):
                    if chr_key == 'q' or chr_key == 'z':
                        do_prev_count = 1 if not shift_pressed else 10
                    elif key == 27: #esc
                        is_exit = True
                        next = True
                        break
                    elif filepath is not None:
                        if chr_key == 'e':
                            do_save_move_count = 1 if not shift_pressed else 10
                        elif chr_key == 'c':
                            do_save_count = 1 if not shift_pressed else 10
                        elif chr_key == 'w':
                            do_skip_move_count = 1 if not shift_pressed else 10
                        elif chr_key == 'x':
                            do_skip_count = 1 if not shift_pressed else 10
                            
            if do_prev_count > 0:
                do_prev_count -= 1
                if len(done_paths) > 0:
                    image_paths.insert(0, filepath)                        
                    filepath = done_paths.pop(-1)
                    
                    if filepath.parent != input_path:
                        new_filename_path = input_path / filepath.name
                        filepath.rename ( new_filename_path )                            
                        image_paths.insert(0, new_filename_path)   
                    else:
                        image_paths.insert(0, filepath)   
                        
                    next = True
            elif filepath is not None:
                if do_save_move_count > 0:
                    do_save_move_count -= 1
                    
                    ed.mask_finish()
                    dflimg.embed_and_set (str(filepath), ie_polys=ed.get_ie_polys() )

                    done_paths += [ confirmed_path / filepath.name ]
                    filepath.rename(done_paths[-1])

                    next = True      
                elif do_save_count > 0:
                    do_save_count -= 1
                        
                    ed.mask_finish()
                    dflimg.embed_and_set (str(filepath), ie_polys=ed.get_ie_polys() )
                    
                    done_paths += [filepath]                         
                    
                    next = True         
                elif do_skip_move_count > 0:
                    do_skip_move_count -= 1
                    
                    done_paths += [skipped_path / filepath.name]
                    filepath.rename(done_paths[-1])
                                        
                    next = True         
                elif do_skip_count > 0:
                    do_skip_count -= 1
                        
                    done_paths += [filepath]     
                                        
                    next = True
            else:
                do_save_move_count = do_save_count = do_skip_move_count = do_skip_count = 0
                
            if do_prev_count + do_save_move_count + do_save_count + do_skip_move_count + do_skip_count == 0:
                if ed.switch_screen_changed():
                    io.show_image (wnd_name, ed.make_screen() )

            
        io.process_messages(0.005)
        
    io.destroy_all_windows()    

