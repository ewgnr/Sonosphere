import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"motion_seq": None,
          "synthesis": None,
          "gui": None,
          "latent_dim": 32,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.motion_seq = config["motion_seq"]
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.latent_dim = config["latent_dim"]
        self.ip = config["ip"]
        self.port = config["port"]
        
         
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/seq1index", self.setSeq1Index)
        self.dispatcher.map("/mocap/seq2index", self.setSeq2Index)
        
        self.dispatcher.map("/mocap/seq1frameindex", self.setSeq1FrameIndex)
        self.dispatcher.map("/mocap/seq2frameindex", self.setSeq2FrameIndex)
        
        self.dispatcher.map("/mocap/seq1framerange", self.setSeq1FrameRange)
        self.dispatcher.map("/mocap/seq2framerange", self.setSeq2FrameRange)
        
        self.dispatcher.map("/mocap/seq1frameincr", self.setSeq1FrameIncrement)
        self.dispatcher.map("/mocap/sew2frameincr", self.setSeq2FrameIncrement)        
        
        self.dispatcher.map("/synth/encodingmix", self.setEncodingMix)
        self.dispatcher.map("/synth/encodingoffset", self.setEncodingOffset)    
    
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def setSeq1Index(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeq1Index(index)
        
    def setSeq2Index(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeq2Index(index)
        
    def setSeq1FrameIndex(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setSeq1FrameIndex(index)

    def setSeq2FrameIndex(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setSeq2FrameIndex(index)
        
    def setSeq1FrameRange(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setSeq1FrameRange(startFrame, endFrame)

    def setSeq2FrameRange(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setSeq2FrameRange(startFrame, endFrame)    

    def setSeq1FrameIncrement(self, address, *args):
        
        incr = args[0]
        
        self.synthesis.setSeq1FrameIncrement(incr)

    def setSeq2FrameIncrement(self, address, *args):
        
        incr = args[0]
        
        self.synthesis.setSeq2FrameIncrement(incr)
        
    def setEncodingMix(self, address, *args):
        
        mix = []
        
        for d in range(self.latent_dim):
        
            mix.append(args[d])
        
        self.synthesis.setEncodingMix(mix)
        
    def setEncodingOffset(self, address, *args):
        
        offset = []
        
        for d in range(self.latent_dim):
        
            offset.append(args[d])
        
        self.synthesis.setEncodingOffset(offset)

