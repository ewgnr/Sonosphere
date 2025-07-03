# Sonosphere 

*Sonosphere* is a technical and artistic tool to realize performative works that combine electronic music, contemporary dance, new media, and artificial intelligence. *Sonosphere* brings together several performative principles: it functions as a digital instrument that can be filled with musicians' own sound material, it is an installation inspired by the [Laban Kinesphere](https://en.wikipedia.org/wiki/Space_Harmony) that can be played through body movements in space, it is a human-machine interface where the machine takes on an independent creative role through the use of generative AI, and it is an audiovisual medium in which sonic and visual elements form a unity.

*Sonosphere* consists of several software elements. These are:

- [AudioAnalysis](https://github.com/ewgnr/Sonosphere/tree/main/AudioAnalysis): A small Python script to compute audio descriptors for sound files.
- [AudioClustering](https://github.com/ewgnr/Sonosphere/tree/main/AudioClustering): A Python-based tool for segmenting, clustering and exporting audio segments from sound files based on audio descriptors.
- [AudioClusteringInteractive](https://github.com/ewgnr/Sonosphere/tree/main/AudioClusteringInteractive): Provides the same functionality as AudioClustering but operates in real-time. 
- [AudioCLustering_v2](https://github.com/ewgnr/Sonosphere/tree/main/AudioClusteringV2): Provides the same functionality as AudioClustering but allows to combine multiple audio descriptors for clustering. 
- [MocapPlayer](https://github.com/ewgnr/Sonosphere/tree/main/MocapPlayer): A Python-based tool to play motion capture recordings of solo performers.
- [MotionClusteringInteractive](https://github.com/ewgnr/Sonosphere/tree/main/MotionClusteringInteractive): A Python-based tool for segmenting and clustering motion capture recordings based on motion descriptors in real-time. 
- [MotionContinuation](https://github.com/ewgnr/Sonosphere/tree/main/MotionContinuation): A Python-based tool that employs a generative AI to predict the continuation of dance movements. 
- [MotionTransformation](https://github.com/ewgnr/Sonosphere/tree/main/MotionTransformation): A Python-based tool that employs a generative AI to interpolate between and manipulate dance movements.
- [SphericalSoundfilePlayer_Mac](https://github.com/ewgnr/Sonosphere/tree/main/SphericalSoundfilePlayer_v8_Mac): The MacOS version of a C++-based tool for processing and playing audio files that are distributed on the surface of a virtual sphere
- [SphericalSoundfilePlayer_Windows](https://github.com/ewgnr/Sonosphere/tree/main/SphericalSoundfilePlayer_v8_Win): The Windows version of a C++-based tool for processing and playing audio files that are distributed on the surface of a virtual sphere
- [RayMarching](https://github.com/ewgnr/Sonosphere/tree/main/RayMarching): OpenGL-based tool for generating video based on ray marching method.



## Table of Contents

* <a href="https://github.com/ewgnr/SphericalInstruments/blob/main/README.md#installation">Installation</a>
* <a href="https://github.com/ewgnr/SphericalInstruments/blob/main/README.md#License">License</a>
* <a href="https://github.com/ewgnr/SphericalInstruments/blob/main/README.md#Authors">Authors</a>

## Installation

* **create conda environment**
  
  conda create --name ima2024 python=3.10

* **activate conda environment**
  
  conda activate ima2024

* **add conda forge as additional default channel**
  
  conda config --add channels conda-forge

* **install python packages using conda**
  
  conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 -c pytorch
  
  conda install spyder

  conda install scikit-learn

  conda install scipy

  conda install librosa

  conda install python-sounddevice

  conda install pysoundfile

  conda install transforms3d

  conda install pandas

  conda install matplotlib

  conda install pyqtgraph

  conda install pyopengl

* **install python packages using pip**

  pip install einops

  pip install einops-ext

  pip install simpleaudio

  pip install python-osc

  pip install fbx-2020.3.7-cp310-none-win_amd64.whl

  pip install fbx-2020.3.7-cp310-cp310-macosx_10_15_universal2.whl (macOS)
  
* **Spherical Soundfile Player**
  
* <a href="https://openframeworks.cc/">Open Frameworks</a>
* <a href="https://www.mega-nerd.com/libsndfile/">libsndfile</a>

## License

<a href="https://creativecommons.org/licenses/by/4.0//">CC Licence 4.0 International</a>

This work is licensed under Creative Commons Attribution-Noncommercial 4.0 International

**You are free to:**
* Share - copy and redistribute the material in any medium or format. Adapt - remix, transform, and build upon the material.
* The licensor cannot revoke these freedoms as lang as you follow the license terms.

**Under the following terms:**
* Attribution -You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* Non-commercial - You may not use the material for commercial purposes.
* No additional restrictions - You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

**Notices:**
* You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.
* No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

## Authors
* <a href="https://github.com/bisnad">Daniel Bisig</a>
* <a href="https://github.com/ewgnr">Ephraim Wegner</a>
