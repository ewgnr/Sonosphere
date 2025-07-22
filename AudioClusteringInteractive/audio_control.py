import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"synthesis": None,
          "model": None,
          "ip": "127.0.0.1",
          "port": 9004}

class AudioControl():
    
    def __init__(self, config):
        
        self.synthesis = config["synthesis"]
        self.model = config["model"]
        self.ip = config["ip"]
        self.port = config["port"]
          
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/synth/clusterlabel", self.setClusterLabel)
        self.dispatcher.map("/synth/audiofeature", self.selectAudioFeature)
        
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def setClusterLabel(self, address, *args):
        
        label = args[0]
        
        self.synthesis.setClusterLabel(label)

    def selectAudioFeature(self, address, *args):
        
        featureName = args[0]

        self.synthesis.selectAudioFeature(featureName)        
