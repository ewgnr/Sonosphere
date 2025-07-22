import torch
from torch import nn
import math
from collections import OrderedDict

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN

"""
model architecture
"""

config = {
    "mocap_data": None,
    "features_data": None,
    "mocap_window_length": 20,
    "mocap_window_offset": 10
    }

class Clustering():
    def __init__(self, mocap_data, features_data, mocap_window_length, mocap_window_offset):
        
        self.mocap_data = mocap_data
        self.features_data = features_data
        self.mocap_window_length = mocap_window_length
        self.mocap_window_offset = mocap_window_offset
        
        self.feature_name = list(self.features_data.keys())[0]
        self.cluster_method = None
        self.cluster_labels = None
        
        self.kmeans_cluster_count = 6
        self.kmeans_rand_state = 170
        
        self._create_mocap_excerpts()
        self._create_feature_excerpts()
        
    def select_motion_feature(self, feature_name):
        
        if feature_name in self.features_data:
            
            self.feature_name = feature_name
            self._create_feature_excerpts()

    def _create_mocap_excerpts(self):

        self.mocap_excerpts = []
        
        mocap_length = self.mocap_data.shape[0]
            
        for excerpt_start in np.arange(0, mocap_length - self.mocap_window_length, self.mocap_window_offset):
            
            mocap_excerpt = self.mocap_data[excerpt_start:excerpt_start + self.mocap_window_length, ...]
            self.mocap_excerpts.append(mocap_excerpt)
                
        self.mocap_excerpts = np.array(self.mocap_excerpts, dtype=np.float32)

    def _create_feature_excerpts(self):

        self.feature_excerpts = []
        
        feature = self.features_data[self.feature_name]
        
        mocap_length = self.mocap_data.shape[0]
            
        for excerpt_start in np.arange(0, mocap_length - self.mocap_window_length, self.mocap_window_offset):
            
            feature_excerpt =  feature[excerpt_start:excerpt_start + self.mocap_window_length, ...]
            self.feature_excerpts.append(feature_excerpt)
                
        self.feature_excerpts = np.array(self.feature_excerpts, dtype=np.float32)
        self.feature_excerpts = np.reshape(self.feature_excerpts, (self.feature_excerpts.shape[0], -1))
        
        
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
        self.cluster_labels =  km.fit_predict(self.feature_excerpts)
        
    def get_label_count(self):
        return len(set(self.cluster_labels))
        
    def get_cluster_mocap_excerpts(self, label):
        
        label_count = len(set(self.cluster_labels))
        excerpt_count = self.mocap_excerpts.shape[0]
        
        cluster_mocap_excerpts = []
        
        for eI in range(excerpt_count):
            
            if self.cluster_labels[eI] == label:
                cluster_mocap_excerpts.append(self.mocap_excerpts[eI])
        
        if len(cluster_mocap_excerpts) > 0:
            cluster_mocap_excerpts = np.stack(cluster_mocap_excerpts, axis=0)    
            return cluster_mocap_excerpts
        else:
            return None
        
        

def createModel(config):
    
    clustering = Clustering(config["mocap_data"], config["features_data"], config["mocap_window_length"], config["mocap_window_offset"])
    clustering.create_cluster_kmeans(config["cluster_count"], config["cluster_random_state"])
    
    return clustering