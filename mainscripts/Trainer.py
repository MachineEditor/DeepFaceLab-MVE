import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from utils import Path_utils
from utils import image_utils    
import cv2
import models
from interact import interact as io

def trainerThread (s2c, c2s, args, device_args):
    while True:
        try: 
            training_data_src_path = Path( args.get('training_data_src_dir', '') )
            training_data_dst_path = Path( args.get('training_data_dst_dir', '') )
            model_path = Path( args.get('model_path', '') )
            model_name = args.get('model_name', '')
            save_interval_min = 15            
            debug = args.get('debug', '')
            
            if not training_data_src_path.exists():
                io.log_err('Training data src directory does not exist.')
                break
                
            if not training_data_dst_path.exists():
                io.log_err('Training data dst directory does not exist.')
                break
                
            if not model_path.exists():
                model_path.mkdir(exist_ok=True)
   
            model = models.import_model(model_name)(
                        model_path, 
                        training_data_src_path=training_data_src_path, 
                        training_data_dst_path=training_data_dst_path, 
                        debug=debug,
                        device_args=device_args)
            
            is_reached_goal = model.is_reached_iter_goal()
            is_upd_save_time_after_train = False
            loss_string = ""
            def model_save():
                if not debug and not is_reached_goal:
                    io.log_info ("Saving....", end='\r')
                    model.save()
                    io.log_info(loss_string)
                    is_upd_save_time_after_train = True
            
            def send_preview():
                if not debug:                        
                    previews = model.get_previews()                
                    c2s.put ( {'op':'show', 'previews': previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
                else:
                    previews = [( 'debug, press update for new', model.debug_one_iter())]
                    c2s.put ( {'op':'show', 'previews': previews} )
            
            
            if model.is_first_run():
                model_save()
                
            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('Model already trained to target iteration. You can use preview.')
                else:
                    io.log_info('Starting. Target iteration: %d. Press "Enter" to stop training and save model.' % ( model.get_target_iter()  ) )
            else: 
                io.log_info('Starting. Press "Enter" to stop training and save model.')
 
            last_save_time = time.time()
            
            for i in itertools.count(0,1):
                if not debug:
                    if not is_reached_goal:
                        loss_string = model.train_one_iter()     
                        if is_upd_save_time_after_train:
                            #save resets plaidML programs, so upd last_save_time only after plaidML rebuild them
                            last_save_time = time.time()
                        
                        io.log_info (loss_string, end='\r')
                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info ('Reached target iteration.')
                            model_save()
                            is_reached_goal = True
                            io.log_info ('You can use preview now.')

                if not is_reached_goal and (time.time() - last_save_time) >= save_interval_min*60:
                    last_save_time = time.time()
                    model_save()
                    send_preview()
                    
                if i==0:
                    if is_reached_goal:
                        model.pass_one_iter()    
                    send_preview()
                    
                if debug:
                    time.sleep(0.005)
                    
                while not s2c.empty():
                    input = s2c.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'preview':                    
                        if is_reached_goal:
                            model.pass_one_iter()                    
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break
                        
                if i == -1:
                    break
                    
                

            model.finalize()
                
        except Exception as e:
            print ('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    c2s.put ( {'op':'close'} )

   
       
def main(args, device_args):
    io.log_info ("Running trainer.\r\n")
    
    no_preview = args.get('no_preview', False)
    
    s2c = queue.Queue()
    c2s = queue.Queue()
    
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, args, device_args) )
    thread.start()

    if no_preview:
        while True:      
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'close':
                    break
            io.process_messages(0.1)
    else:    
        wnd_name = "Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)
        
        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0    
        iter = 0
        while True:      
            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iter = input['iter'] if 'iter' in input.keys() else 0
                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            (h, w, c) = preview_rgb.shape
                            max_h = max (max_h, h)
                            max_w = max (max_w, w)
                        
                        max_size = 800
                        if max_h > max_size:
                            max_w = int( max_w / (max_h / max_size) )
                            max_h = max_size

                        #make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append ( (preview_name, cv2.resize(preview_rgb, (max_w, max_h))) )
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'close':
                    break
                    
            if update_preview:
                update_preview = False

                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                (h,w,c) = selected_preview_rgb.shape
                
                # HEAD
                head_lines = [
                    '[s]:save [enter]:exit',
                    '[p]:update [space]:next preview [l]:change history range',
                    'Preview: "%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
                    ] 
                head_line_height = 15
                head_height = len(head_lines) * head_line_height
                head = np.ones ( (head_height,w,c) ) * 0.1
                  
                for i in range(0, len(head_lines)):
                    t = i*head_line_height
                    b = (i+1)*head_line_height
                    head[t:b, 0:w] += image_utils.get_text_image (  (w,head_line_height,c) , head_lines[i], color=[0.8]*c )
                    
                final = head
       
                if loss_history is not None:                
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]
                        
                    lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)
                    final = np.concatenate ( [final, lh_img], axis=0 )

                final = np.concatenate ( [final, selected_preview_rgb], axis=0 )
                final = np.clip(final, 0, 1)
                
                io.show_image( wnd_name, (final*255).astype(np.uint8) )
                is_showing = True
            
            key_events = io.get_key_events(wnd_name)
            key, = key_events[-1] if len(key_events) > 0 else (0,)
                    
            if key == ord('\n') or key == ord('\r'):
                s2c.put ( {'op': 'close'} )
            elif key == ord('s'):
                s2c.put ( {'op': 'save'} )
            elif key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put ( {'op': 'preview'} )
            elif key == ord('l'):
                if show_last_history_iters_count == 0:
                    show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                    show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000    
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0                
                update_preview = True
            elif key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True
            
            io.process_messages(0.1)
            
        io.destroy_all_windows()