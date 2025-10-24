"""
OSC Score
"""

"""
Imports
"""

import time
import threading
import json
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

"""
Score Settings
"""

score_progression_auto = True
score_progression_manual_step = False

score = {}

score["part1"] = {}
score["part_sequence"] = ["part1"]

score["part1"]["time"] =    [0,   10,  20,  30,  40 ]
score["part1"]["sound"] =   [100, 101, 102, 102, 300]
score["part1"]["trigger"] = [0,   0,   1,   0,   1  ]
score["part1"]["vis"] =     [401, -1,  402, -1,  -1 ]
score["part1"]["mc"] =      [201, -1,  101, -1,  -1 ]

"""
OSC Settings
"""

osc_settings = {}

osc_settings["sound"] = {}
osc_settings["vis"] = {}
osc_settings["mc"] = {}

osc_settings["sound"]["ip"] = "127.0.0.1"
osc_settings["sound"]["port"] = 9005
osc_settings["sound"]["ma"] = "/audio/preset"

osc_settings["vis"]["ip"] = "127.0.0.1"
osc_settings["vis"]["port"] = 9006
osc_settings["vis"]["ma"] = "/vis/preset"

osc_settings["mc"]["ip"] = "127.0.0.1"
osc_settings["mc"]["port"] = 9007
osc_settings["mc"]["ma"] = "/mc/preset"

"""
OSC Functions
"""

def setup_osc_senders():
    
    for key in osc_settings:
        osc_settings[key]["sender"] = SimpleUDPClient(osc_settings[key]["ip"], osc_settings[key]["port"])

def score_osc_send(score_part_name, score_step):
    
    # send audio preset
    osc_sender = osc_settings["sound"]["sender"]
    osc_ma = osc_settings["sound"]["ma"]
    preset_nr = score[score_part_name]["sound"][score_step]
    trigger = score[score_part_name]["trigger"][score_step]
    
    if preset_nr != -1:
        osc_sender.send_message(osc_ma, [preset_nr, trigger]) 

    # send vis preset
    osc_sender = osc_settings["vis"]["sender"]
    osc_ma = osc_settings["vis"]["ma"]
    preset_nr = score[score_part_name]["vis"][score_step]
    
    if preset_nr != -1:
        osc_sender.send_message(osc_ma, [preset_nr]) 
        
    # send mc preset
    osc_sender = osc_settings["mc"]["sender"]
    osc_ma = osc_settings["mc"]["ma"]
    preset_nr = score[score_part_name]["mc"][score_step]
    
    if preset_nr != -1:
        osc_sender.send_message(osc_ma, [preset_nr]) 


setup_osc_senders()

"""
Score Functions
"""

current_score_part = 0
current_score_step = 0 

"""
Score Progression
"""

score_update_interval = 0.01 # seconds

stop_event = threading.Event()

def run_score_task():
    
    global stop_event
    global score_thread
    global current_score_part
    global current_score_step
    global score_progression_manual_step
    
    start_time = time.perf_counter()
    
    while not stop_event.is_set():
        
        elapsed = time.perf_counter() - start_time
        
        score_part_name = score["part_sequence"][current_score_part]
        score_step_time = score[score_part_name]["time"][current_score_step]
        
        if (score_progression_auto == True and elapsed > score_step_time) or (score_progression_auto == False and score_progression_manual_step == True):
           
            score_progression_manual_step = False
            
            print("part ", score_part_name, " step ", current_score_step, " step_time",score_step_time, " elapsed ",  elapsed)
            #print("elapsed ", elapsed, " score_step_time ", score_step_time, " current_score_step ", current_score_step)
            
            score_osc_send(score_part_name, current_score_step)

            current_score_step += 1
            
            if current_score_step >= len(score[score_part_name]["time"]):
                
                current_score_step = 0
                current_score_part += 1
                start_time = time.perf_counter()
                
                if current_score_part >= len(score["part_sequence"]):

                    stop_event.set()
                    
        
        # Wait until 10 ms have passed since loop start
        sleep_duration = max(0, score_update_interval - elapsed)
        time.sleep(sleep_duration)

def start_score_task():
    score_thread = threading.Thread(target=run_score_task, args=())
    score_thread.start()
    return score_thread

score_thread = start_score_task()




"""
score_thread.join()
"""