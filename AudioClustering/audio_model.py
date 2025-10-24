import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN

"""
model architecture
"""

config = {
    "audio_excerpts": None,
    "audio_features": None,
    "cluster_method": "kmeans",
    "cluster_count": 20,
    "cluster_random_state": 170
    }

class Clustering():
    def __init__(self, audio_excerpts, audio_features):
        
        self.audio_excerpts = audio_excerpts
        self.audio_features = audio_features

        self.feature_name = list(self.audio_features.keys())[0]
        self.cluster_method = None
        self.cluster_labels = None
        
        self.kmeans_cluster_count = 20
        self.kmeans_rand_state = 170

    def select_audio_feature(self, feature_name):
        
        print("select_audio_feature ", feature_name)
        
        if feature_name in self.audio_features:
            
            self.feature_name = feature_name
            self.create_clusters()       
        
    def create_clusters(self):
        
        if self.cluster_method == "kmeans":
            self._create_clusters_kmeans()
            
    def create_cluster_kmeans(self, cluster_count, rand_state):
        
        self.cluster_method = "kmeans"
        self.kmeans_cluster_count  = cluster_count
        self.kmeans_rand_state  = rand_state
        
        self._create_clusters_kmeans()

    def _create_clusters_kmeans(self):
        
        km = KMeans(n_clusters=self.kmeans_cluster_count, n_init= "auto", random_state = self.kmeans_rand_state)
        self.cluster_labels =  km.fit_predict(self.audio_features[self.feature_name])
        
    def get_label_count(self):
        return len(set(self.cluster_labels))
        
    def get_cluster_audio_excerpts(self, label):
        
        label_count = len(set(self.cluster_labels))
        
        excerpt_count = self.audio_excerpts.shape[0]
        
        cluster_audio_excerpts = []
        
        for eI in range(excerpt_count):
            
            if self.cluster_labels[eI] == label:
                cluster_audio_excerpts.append(self.audio_excerpts[eI])
        
        if len(cluster_audio_excerpts) > 0:
            cluster_audio_excerpts = np.stack(cluster_audio_excerpts, axis=0)    
            return cluster_audio_excerpts
        else:
            return None
        
        

def createModel(config):
    
    clustering = Clustering(config["audio_excerpts"], config["audio_features"])
    clustering.create_cluster_kmeans(config["cluster_count"], config["cluster_random_state"])
    
    return clustering