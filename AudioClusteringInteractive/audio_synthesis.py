import numpy as np

config = {"model": None,
          "audio_excerpts": None,
          "audio_sample_rate ": 48000,
          "audio_excerpt_length": 1000,
          "audio_excerpt_offset": 900
          }


"""
audio_excerpt_length_sc = int(audio_excerpt_length / 1000 * audio_sample_rate)
audio_excerpt_offset_sc = int(audio_excerpt_offset / 1000 * audio_sample_rate)
"""

class AudioSynthesis():
    
    def __init__(self, config):
        self.model = config["model"]
        self.audio_sample_rate = config["audio_sample_rate"]
        self.audio_excerpt_length = config["audio_excerpt_length"]
        self.audio_excerpt_offset = config["audio_excerpt_offset"]
        
        self.audio_excerpt_length_sc = int(self.audio_excerpt_length / 1000 * self.audio_sample_rate)
        self.audio_excerpt_offset_sc = int(self.audio_excerpt_offset / 1000 * self.audio_sample_rate)
        
        self._create_envelope()
        
        self.cluster_label = 0
        self.cluster_waveform = None
        
        self.play_sample_index = 0
        
        self.cluster_audio = None
        
    def _create_envelope(self):
        
        overlap_sc = self.audio_excerpt_length_sc - self.audio_excerpt_offset_sc

        env_part1 = np.linspace(0.0 ,1.0, overlap_sc)
        env_part2 = np.ones(self.audio_excerpt_length_sc - 2 * overlap_sc, dtype=np.float32)
        env_part3 = np.linspace(1.0 ,0.0, overlap_sc)
        
        self.audio_window_envelope = np.concatenate((env_part1, env_part2, env_part3))
        
    def _create_cluster_audio(self):
        
        cluster_audio_excerpts = self.model.get_cluster_audio_excerpts(self.cluster_label)
        
        excerpt_count = cluster_audio_excerpts.shape[0]

        if excerpt_count == 0:
            self.cluster_audio = None
        else:
            audio_cluster_sc = self.audio_excerpt_length_sc + self.audio_excerpt_offset_sc * (excerpt_count - 1)
            self.cluster_audio = np.zeros(audio_cluster_sc, dtype=np.float32)

        audio_cluster_insert_index = 0  
            
        for audio_excerpt in cluster_audio_excerpts:
            
            self.cluster_audio[audio_cluster_insert_index:audio_cluster_insert_index+self.audio_excerpt_length_sc] += audio_excerpt * self.audio_window_envelope
            
            audio_cluster_insert_index += self.audio_excerpt_offset_sc

    def setClusterLabel(self, label):
        
        self.cluster_label = min(label, self.model.get_label_count() - 1)
        
        self.play_sample_index = 0
        
        self._create_cluster_audio()
        
    def selectAudioFeature(self, fleatureName):
        
        self.model.select_audio_feature(fleatureName)
        self.model.create_clusters()
        
        self.setClusterLabel(0)      
            
    def update(self, audio_buffer):
        
        if self.cluster_audio is None:
            return
        
        audio_buffer_length = audio_buffer.shape[0]
        cluster_audio_length = self.cluster_audio.shape[0]
        
        #print("audio_buffer_length ", audio_buffer_length)
        #print("cluster_audio_length ", cluster_audio_length)
        
        #print("self.play_sample_index ", self.play_sample_index, " cluster_audio_length ", cluster_audio_length)
        
        if self.play_sample_index + audio_buffer_length < cluster_audio_length:
            
            audio_buffer[:] = self.cluster_audio[self.play_sample_index:self.play_sample_index+audio_buffer_length]
            
            #print("audio_buffer ", audio_buffer)
            
            self.play_sample_index += audio_buffer_length
        else:
            
            buffer_part1_length = cluster_audio_length - self.play_sample_index
            
            audio_buffer[:buffer_part1_length] = self.cluster_audio[self.play_sample_index:self.play_sample_index+buffer_part1_length]
            
            self.play_sample_index = 0
            
            buffer_part2_length = audio_buffer_length - buffer_part1_length
            
            audio_buffer[buffer_part1_length:] = self.cluster_audio[0:buffer_part2_length]
            
            self.play_sample_index = buffer_part2_length

        