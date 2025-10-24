# osc_score_pyqt_minimal.py

import sys
import time
import threading
from pythonosc.udp_client import SimpleUDPClient
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox
)
from PyQt5.QtCore import QTimer

# ---------- Score Definition ----------

score_progression_auto = True
score_progression_manual_step = False

score = {
    "part1": {
        "time":    [0,   1,   60,  120, 121, 123, 180, 181, 183, 210, 211, 212, 213, 214, 240, 270, 271, 273, 330, 331 ],
        "sound":   [100, 100, 101, 101, 102, 102, 102, 300, 300, 301, 300, 301, 300, 302, 303, 303, 700, 700, 700, 700 ],
        "trigger": [0,   1,   1,   0,   0,   1,   0,   0,   1,   1,   1,   1,   1,   1,   1,   0,   0,   1,   1,   0   ],
        "swarm":   [100, 100, 102, 102, 102, 102, 300, 300, 300, 300, 300, 300, 300, 300, 300, 700, 700, 700, 30,  30  ]
    },
    "part_sequence": ["part1"]
}
"""
score = {
    "part1": {
        "time":    [0,   1,   60,  120, 121, 123, 180, 181, 183, 240, 241, 243, 300, 301 ],
        "sound":   [100, 100, 101, 101, 102, 102, 102, 300, 300, 300, 700, 700, 700, 700 ],
        "trigger": [0,   1,   1,   0,   0,   1,   0,   0,   1,   0,   0,   1,   1,   0   ],
        "swarm":   [100, 100, 102, 102, 102, 102, 300, 300, 300, 700, 700, 700, 30,  30  ]
    },
    "part_sequence": ["part1"]
}
"""
# ---------- OSC Settings ----------

osc_settings = {
    "sound": {
        "ip": "127.0.0.1",
        "port": 9004,
        "ma": "/audio/preset"
    },
    "swarm": {
        "ip": "127.0.0.1",
        "port": 9003,
        "ma": "/swarm/preset"
    }
}

# ---------- OSC Functions ----------

def setup_osc_senders():
    for key in osc_settings:
        settings = osc_settings[key]
        settings["sender"] = SimpleUDPClient(settings["ip"], settings["port"])

def score_osc_send(score_part_name, score_step):
    part = score[score_part_name]

    # Sound
    sound = part["sound"][score_step]
    trigger = part["trigger"][score_step]
    if sound != -1:
        osc_settings["sound"]["sender"].send_message(osc_settings["sound"]["ma"], [sound, trigger])

    # Swarm
    swarm = part["swarm"][score_step]
    if swarm != -1:
        osc_settings["swarm"]["sender"].send_message(osc_settings["swarm"]["ma"], [swarm])

setup_osc_senders()

# ---------- Score Playback ----------

current_score_part = 0
current_score_step = 0
score_update_interval = 0.01
stop_event = threading.Event()
score_thread = None

def run_score_task():
    global stop_event, current_score_part, current_score_step, score_progression_manual_step

    start_time = time.perf_counter()
    while not stop_event.is_set():
        elapsed = time.perf_counter() - start_time
        part_name = score["part_sequence"][current_score_part]
        step_time = score[part_name]["time"][current_score_step]

        if (score_progression_auto and elapsed > step_time) or \
           (not score_progression_auto and score_progression_manual_step):
            score_progression_manual_step = False
            print(f"Part: {part_name}, Step: {current_score_step}, Time: {step_time}, Elapsed: {elapsed:.2f}")
            score_osc_send(part_name, current_score_step)

            current_score_step += 1
            if current_score_step >= len(score[part_name]["time"]):
                current_score_step = 0
                current_score_part += 1
                start_time = time.perf_counter()
                if current_score_part >= len(score["part_sequence"]):
                    stop_event.set()
        time.sleep(score_update_interval)

def start_score_task():
    global stop_event, score_thread
    stop_event.clear()
    score_thread = threading.Thread(target=run_score_task)
    score_thread.start()

# ---------- GUI ----------

class ScoreGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OSC Score Control')
        self.layout = QVBoxLayout()

        self.display_labels = {}
        for label in ['Score Part', 'Step Index', 'Time', 'Sound', 'Trigger', 'Swarm']:
            lbl = QLabel(f"{label}:")
            self.display_labels[label] = lbl
            self.layout.addWidget(lbl)

        # Controls
        self.start_btn = QPushButton('Start')
        self.stop_btn = QPushButton('Stop')
        self.reset_btn = QPushButton('Reset')
        self.auto_checkbox = QCheckBox('Automated Progression')
        self.step_btn = QPushButton('Step (Manual)')

        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.reset_btn.clicked.connect(self.on_reset)
        self.auto_checkbox.setChecked(score_progression_auto)
        self.auto_checkbox.stateChanged.connect(self.on_toggle_auto)
        self.step_btn.clicked.connect(self.on_step)

        ctrl_layout = QHBoxLayout()
        for btn in [self.start_btn, self.stop_btn, self.reset_btn, self.auto_checkbox, self.step_btn]:
            ctrl_layout.addWidget(btn)
        self.layout.addLayout(ctrl_layout)

        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(150)

    def on_start(self):
        global score_thread
        if not score_thread or not score_thread.is_alive():
            start_score_task()

    def on_stop(self):
        global stop_event, score_thread
        stop_event.set()
        if score_thread and score_thread.is_alive():
            score_thread.join()
            score_thread = None

    def on_reset(self):
        global stop_event, current_score_part, current_score_step, score_thread
        stop_event.set()
        if score_thread and score_thread.is_alive():
            score_thread.join()
            score_thread = None
        current_score_part = 0
        current_score_step = 0

    def on_toggle_auto(self):
        global score_progression_auto
        score_progression_auto = self.auto_checkbox.isChecked()

    def on_step(self):
        global score_progression_manual_step
        score_progression_manual_step = True

    def update_display(self):
        global current_score_part, current_score_step

        seq = score["part_sequence"]
        part_name = seq[current_score_part] if current_score_part < len(seq) else "(finished)"
        part = score.get(part_name, {})
        idx = current_score_step

        get_val = lambda key: part.get(key, [""])[idx] if key in part and idx < len(part[key]) else ""

        self.display_labels['Score Part'].setText(f"Score Part: {part_name}")
        self.display_labels['Step Index'].setText(f"Step Index: {idx}")
        self.display_labels['Time'].setText(f"Time: {get_val('time')}")
        self.display_labels['Sound'].setText(f"Sound: {get_val('sound')}")
        self.display_labels['Trigger'].setText(f"Trigger: {get_val('trigger')}")
        self.display_labels['Swarm'].setText(f"Swarm: {get_val('swarm')}")

# ---------- Main ----------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScoreGUI()
    window.show()
    sys.exit(app.exec_())
