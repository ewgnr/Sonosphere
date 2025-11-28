"""
Audio Autoencoder (CNN Version)
"""

"""
Imports
"""

import os
import math
import numpy as np
import threading
import queue
import torch
from torch import nn
import torchaudio
from torchaudio.functional import highpass_biquad
import sounddevice as sd
from vocos import Vocos
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys
from matplotlib import pyplot as plt

"""
Settings
"""

"""
Device Settings
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device.upper()} device")

"""
Audio Settings
"""

audio_file = "data/audio/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav"
audio_sample_rate = 48000
audio_channel_count = 1
audio_window_length = 2048
audio_output_device = 7 # windows: 7, macOS: 2
max_audio_queue_length = 32

# automatically calculated settings
gen_buffer_size = audio_window_length
window_size = gen_buffer_size
window_offset = window_size // 2
play_buffer_size = window_size * 16
playback_latency = 1.0

"""
Autoencoder Settings
"""

latent_dim = 32
ae_conv_channel_counts = [ 16, 32, 64, 128 ]
ae_conv_kernel_size = (5, 3)
ae_dense_layer_sizes = [ 512 ]
ae_encoder_weights_file = "data/results/weights/audio_vocos_cnn_encoder_weights_epoch_400"
ae_decoder_weights_file = "data/results/weights/audio_vocos_cnn_decoder_weights_epoch_400"

"""
Load Audio
"""

assert os.path.exists(audio_file), f"Audio file not found: {audio_file}"

audio_waveform, _ = torchaudio.load(audio_file)
audio_source_samples = audio_waveform[0].to(device)
audio_source_frame_index = 0
hann_window = torch.from_numpy(np.hanning(window_size)).float().to(device) # Move to device once

"""
Audio Excerpt Generation for 2D Mapping
"""

audio_excerpt_frame_offset = 10000

audio_excerpt_start_frame = 0
audio_excerpt_end_frame = audio_waveform.shape[1]
audio_excerpts = []
for fI in range(audio_excerpt_start_frame, audio_excerpt_end_frame - gen_buffer_size, audio_excerpt_frame_offset):
    audio_excerpt = audio_waveform[0, fI:fI + gen_buffer_size]
    audio_excerpts.append(audio_excerpt)
audio_excerpts = torch.stack(audio_excerpts, dim=0)

"""
Create Models
"""

"""
Create Vocoder Model
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()

"""
Determine Number of Mel Filters and Mel Spectra
"""

with torch.no_grad():
    dummy_features = vocos.feature_extractor(torch.rand(size=(1, gen_buffer_size), device=device))
    mel_count = dummy_features.shape[-1]
    mel_filters = dummy_features.shape[1]
    
"""
Create VAE Encoder
"""

class Encoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)

        padding = stride
        
        self.conv_layers.append(nn.Conv2d(1, conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding))
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index]))

        self.flatten = nn.Flatten()
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))

        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size_x, last_conv_layer_size_y]
        
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y

        self.dense_layers.append(nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        # create final dense layers
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)


    def forward(self, x):
        for lI, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = self.flatten(x)
        for lI, layer in enumerate(self.dense_layers):
            x = layer(x)
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        return mu, std
    
    @staticmethod
    def reparameterize(mu, std):
        z = mu + std*torch.randn_like(std)
        return z

encoder = Encoder(latent_dim, mel_count, mel_filters, ae_conv_channel_counts, ae_conv_kernel_size, ae_dense_layer_sizes).to(device)
encoder.load_state_dict(torch.load(ae_encoder_weights_file, map_location=device))
encoder.eval()

"""
Create VAE Decoder
"""

class Decoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filters, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filters = mel_filters
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)

        self.dense_layers.append(nn.Linear(latent_dim, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        last_conv_layer_size_x = int(mel_filters // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))

        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]

        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y
 
        self.dense_layers.append(nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size))
        self.dense_layers.append(nn.ReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        padding = stride
        output_padding = (padding[0] - 1, padding[1] - 1) # does this universally work?
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))

    def forward(self, x):

        for lI, layer in enumerate(self.dense_layers):
            x = layer(x)
        x = self.unflatten(x)
        for lI, layer in enumerate(self.conv_layers):
            x = layer(x)

        return x

decoder = Decoder(latent_dim, mel_count, mel_filters, list(reversed(ae_conv_channel_counts)), ae_conv_kernel_size, list(reversed(ae_dense_layer_sizes))).to(device)
decoder.load_state_dict(torch.load(ae_decoder_weights_file, map_location=device))
decoder.eval()

"""
Generate latent encodings and 2D projections
"""

batch_size = 32

with torch.no_grad():
    audio_encodings = []
    for eI in range(0, audio_excerpts.shape[0] - batch_size, batch_size):
        batch = audio_excerpts[eI:eI+batch_size].to(device)
        mels = vocos.feature_extractor(batch)
        audio_encoder_in = mels.unsqueeze(1)
        mu, std = encoder(audio_encoder_in)
        std = torch.nn.functional.softplus(std) + 1e-6
        encoded_batch = Encoder.reparameterize(mu, std).detach().cpu()
        audio_encodings.append(encoded_batch)
    audio_encodings = torch.cat(audio_encodings, dim=0).numpy()
tsne = TSNE(n_components=2, max_iter=5000, verbose=1)
Z_tsne = tsne.fit_transform(audio_encodings)

"""
Display 2D Projection
"""

Z_tsne_x = Z_tsne[:,0]
Z_tsne_y = Z_tsne[:,1]

plt.figure()
fig, ax = plt.subplots()
ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=1.0)

"""
K-Nearest-Neighbors Search based on Mouse Clicks
"""

n_neighbors = 4

knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
knn.fit(Z_tsne)

def calc_distance_based_averaged_encoding(point2D):
    """Returns distance-weighted averaged encoding for a given 2D point."""
    _, indices = knn.kneighbors(point2D)
    nearest_positions = Z_tsne[indices[0]]
    nearest_encodings = audio_encodings[indices[0]]
    nearest_2D_distances = np.linalg.norm(nearest_positions - point2D, axis=1)
    max_2D_distance = np.max(nearest_2D_distances)
    norm_nearest_2D_distances = nearest_2D_distances / max_2D_distance
    weights = (1.0 - norm_nearest_2D_distances)
    return np.average(nearest_encodings, weights=weights, axis=0)

"""
Real-Time Audio Callback Functions
"""

inter_audio_encodings = []
inter_audio_encoding_index = 0

@torch.no_grad()
def encode_audio(waveform):
    waveform = waveform.unsqueeze(0).to(device)
    mels = vocos.feature_extractor(waveform)
    mu, std = encoder(mels.unsqueeze(1))
    std = torch.nn.functional.softplus(std) + 1e-6
    return Encoder.reparameterize(mu, std)

@torch.no_grad()
def decode_audio(latent):
    mel_pred = decoder(latent).squeeze(1)
    waveform = vocos.decode(mel_pred).reshape(1, -1)
    return waveform

@torch.no_grad()
def synthesize_audio():
    """Decode the next latent encoding from the current interactive list."""
    global inter_audio_encoding_index
    if len(inter_audio_encodings) == 0:
        return torch.zeros(gen_buffer_size)
    inter_audio_encoding_index += 1
    if inter_audio_encoding_index >= len(inter_audio_encodings):
        inter_audio_encoding_index = 0
    latent = inter_audio_encodings[inter_audio_encoding_index]
    gen_waveform = decode_audio(latent).reshape(-1)
    return gen_waveform

"""
Audio Threading
"""

audio_queue = queue.Queue(maxsize=max_audio_queue_length)
export_audio_buffer = []
last_chunk = np.zeros(window_size, dtype=np.float32)

def producer_thread():
    """Continuously generates audio and fills the output buffer queue."""
    while True:
        if not audio_queue.full():
            gen_waveform = synthesize_audio()
            audio_queue.put(gen_waveform.cpu().numpy())
        else:
            sd.sleep(10)

def audio_callback(out_data, frames, time_info, status):
    """sounddevice stream callback function."""
    global last_chunk
    output = np.zeros((frames, audio_channel_count), dtype=np.float32)
    cursor = 0
    overlap_len = window_size // 2
    output[cursor:cursor+overlap_len, 0] += last_chunk[overlap_len:]
    samples_needed = frames
    while samples_needed > 0:
        try:
            chunk = audio_queue.get_nowait()
            chunk = (chunk * hann_window.cpu().numpy())
        except queue.Empty:
            chunk = np.zeros(window_size, dtype=np.float32)
        chunk_copy_size = output[cursor:cursor+window_size, 0].shape[0]
        output[cursor:cursor+chunk_copy_size, 0] += chunk[:chunk_copy_size]
        cursor += window_size // 2
        samples_needed = frames - cursor
        last_chunk[:] = chunk[:]
    out_data[:] = output

def run_audio_stream():
    """Main thread for launching audio producer and output stream."""
    threading.Thread(target=producer_thread, daemon=True).start()
    sd.sleep(2000)
    with sd.OutputStream(
        samplerate=audio_sample_rate,
        device=audio_output_device,
        channels=audio_channel_count,
        callback=audio_callback,
        blocksize=play_buffer_size,
        latency=playback_latency
    ):
        print("Streaming audio with FIFO queue... press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped.")
            
"""
Scatter Plot Application
"""

class ScatterPlotApp(QtWidgets.QMainWindow):
    """
    Interactive scatter plot GUI for latent encodings.
    - Left button press/drag: Add points and encodings.
    - Right button press/drag: Remove points and corresponding encodings near cursor.
    - Middle button: pan plot.
    - 'C' key: clear all points/encodings.
    """
    def __init__(self, points2D, inter_audio_encodings):
        super().__init__()
        self.setWindowTitle("2D Scatter Plot with Click and Drag")
        # Container and layout for pyqtgraph widget
        container_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container_widget)
        self.graph_widget = pg.PlotWidget()
        layout.addWidget(self.graph_widget)
        self.setCentralWidget(container_widget)

        # Main scatter item (fixed background points)
        self.scatter = pg.ScatterPlotItem(
            x=points2D[:, 0],
            y=points2D[:, 1],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            size=10
        )
        self.graph_widget.addItem(self.scatter)

        # Interactive points and encodings (red points are user-selected)
        self.click_points = []
        self.click_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 0, 0, 200))
        self.graph_widget.addItem(self.click_scatter)
        
        # Audio playback point
        self.play_point = []
        self.play_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(0, 255, 0, 200))
        self.graph_widget.addItem(self.play_scatter)

        # Interaction state
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.right_button_pressed = False
        self.last_mouse_pos = None

        # Timer for continuous add (while dragging left button)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)  # ms
        self.timer.timeout.connect(self.continuous_add_point)
        
        # Timer for updating play point
        self.playpoint_timer = QtCore.QTimer()
        self.playpoint_timer.setInterval(25)  # ms
        self.playpoint_timer.timeout.connect(self.update_play_point)
        self.playpoint_timer.start()

        # Register event filter for mouse events
        self.graph_widget.scene().installEventFilter(self)
        self.click_encodings = inter_audio_encodings

    def addInteractiveEncoding(self, mouse_point):
        """Add an encoding for the clicked position."""
        encoding = calc_distance_based_averaged_encoding(np.array([[mouse_point.x(), mouse_point.y()]]))
        encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(device)
        self.click_encodings.append(encoding)

    def removeInteractiveEncodingNear(self, mouse_point, radius=0.5):
        """Remove points and encodings within a radius of the mouse cursor."""
        to_remove_indices = []
        mp_x, mp_y = mouse_point.x(), mouse_point.y()
        for i, p in enumerate(self.click_points):
            dx = p['pos'][0] - mp_x
            dy = p['pos'][1] - mp_y
            dist = (dx*dx + dy*dy)**0.5
            if dist < radius:
                to_remove_indices.append(i)
        # Remove from lists in reverse order for safety
        for idx in reversed(to_remove_indices):
            self.click_points.pop(idx)
            self.click_encodings.pop(idx)
        # Update plot
        self.click_scatter.setData(
            [p['pos'][0] for p in self.click_points],
            [p['pos'][1] for p in self.click_points]
        )

    def clearInteractiveEncodings(self):
        """Clear all interactive points and encodings."""
        self.click_points.clear()
        self.click_scatter.setData([], [])
        self.click_encodings.clear()

    def eventFilter(self, source, event):
        """Handle mouse events for the plot."""
        if source == self.graph_widget.scene():
            if event.type() == QtCore.QEvent.GraphicsSceneMousePress:
                if event.button() == QtCore.Qt.LeftButton:
                    self.left_button_pressed = True
                    self.last_mouse_pos = event.scenePos()
                    self.add_point_at(event.scenePos())
                    self.timer.start()
                    return True
                elif event.button() == QtCore.Qt.MiddleButton:
                    self.middle_button_pressed = True
                    self.last_mouse_pos = event.scenePos()
                    return True
                elif event.button() == QtCore.Qt.RightButton:
                    self.right_button_pressed = True
                    self.last_mouse_pos = event.scenePos()
                    return True
            elif event.type() == QtCore.QEvent.GraphicsSceneMouseRelease:
                if event.button() == QtCore.Qt.LeftButton:
                    self.left_button_pressed = False
                    self.timer.stop()
                    return True
                elif event.button() == QtCore.Qt.MiddleButton:
                    self.middle_button_pressed = False
                    return True
                elif event.button() == QtCore.Qt.RightButton:
                    self.right_button_pressed = False
                    return True
            elif event.type() == QtCore.QEvent.GraphicsSceneMouseMove:
                if self.left_button_pressed:
                    self.last_mouse_pos = event.scenePos()
                    return True
                elif self.middle_button_pressed:
                    if self.last_mouse_pos is not None:
                        diff = event.scenePos() - self.last_mouse_pos
                        vb = self.graph_widget.plotItem.vb
                        vb.translateBy(x=-diff.x(), y=diff.y())
                        self.last_mouse_pos = event.scenePos()
                    return True
                elif self.right_button_pressed:
                    # Remove points near mouse position on drag
                    vb = self.graph_widget.plotItem.vb
                    mouse_point = vb.mapSceneToView(event.scenePos())
                    self.removeInteractiveEncodingNear(mouse_point)
                    self.last_mouse_pos = event.scenePos()
                    return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        """Clear points/encodings on 'C' key."""
        if event.key() == QtCore.Qt.Key_C:
            self.clearInteractiveEncodings()
        super().keyPressEvent(event)

    def add_point_at(self, scene_pos):
        """Add a scatter point and encoding at mouse position."""
        if self.graph_widget.sceneBoundingRect().contains(scene_pos):
            vb = self.graph_widget.plotItem.vb
            mouse_point = vb.mapSceneToView(scene_pos)
            self.addInteractiveEncoding(mouse_point)
            x, y = mouse_point.x(), mouse_point.y()
            self.click_points.append({'pos': (x, y)})
            self.click_scatter.setData(
                [p['pos'][0] for p in self.click_points],
                [p['pos'][1] for p in self.click_points]
            )

    def continuous_add_point(self):
        """Timer-driven continuous interactive addition while dragging."""
        if self.left_button_pressed and self.last_mouse_pos is not None:
            self.add_point_at(self.last_mouse_pos)
            
    def update_play_point(self):
        
        global inter_audio_encoding_index
        
        if inter_audio_encoding_index >= len(self.click_points):
            return
        
        play_p = self.click_points[inter_audio_encoding_index]

        self.play_scatter.setData( [ play_p['pos'][0] ], [ play_p['pos'][1] ] )

"""
Run Application
"""

# Run audio stream in separate thread for GUI/audio concurrency
audio_thread = threading.Thread(target=run_audio_stream, daemon=True)
audio_thread.start()
app = QtWidgets.QApplication(sys.argv)
main = ScatterPlotApp(Z_tsne, inter_audio_encodings)
main.show()
sys.exit(app.exec_())
