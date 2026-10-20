# Sonosphere

*Sonosphere* is a technical and artistic platform for creating performative works that combine electronic music, contemporary dance, new media, swarm simulation, and artificial intelligence. *Sonosphere* brings together several performative principles: it functions as a digital instrument that can be filled with musicians' own sound material; it is an installation inspired by the [Laban Kinesphere](https://en.wikipedia.org/wiki/Space_Harmony) that can be played through body movements in space; it uses Azure Kinect motion tracking to capture performers' movements; it incorporates a swarm simulation whose autonomous agents can influence the audiovisual environment; it is a human-machine interface in which computational processes and generative AI can take on an independent creative role; and it is an audiovisual medium in which sonic and visual elements form a unity.

*Sonosphere* consists of several interconnected software elements. These are:

- **Azure Kinect Tracker:** Captures and processes real-time body movement data for interaction with Sonosphere.
- **Swarm Simulation:** Simulates autonomous agents whose collective, emergent behaviour can interact with performers and influence sonic and visual parameters.
- [SphericalSoundfilePlayer_Mac](https://github.com/ewgnr/Sonosphere/tree/main/SphericalSoundfilePlayer_v8_Mac): The macOS version of a C++-based tool for processing and playing audio files distributed on the surface of a virtual sphere.
- [SphericalSoundfilePlayer_Windows](https://github.com/ewgnr/Sonosphere/tree/main/SphericalSoundfilePlayer_v8_Win): The Windows version of the spherical sound instrument.
- [RayMarching](https://github.com/ewgnr/Sonosphere/tree/main/RayMarching): An OpenGL-based tool for generating video using the ray marching method.

Python-based machine learning, audio analysis, motion processing, and generative AI tools are maintained in the separate [CAS_AISoundMotion_2026](https://github.com/bisnad/CAS_AISoundMotion_2026) repository. These tools can be used independently to prepare and process audio and movement material, **or connected to Sonosphere to interact with it**, including during real-time performances. The separate repositories allow the tools to be maintained independently while remaining part of the broader Sonosphere ecosystem.

## Table of Contents

- [Software Components](#software-components)
- [Installation](#installation)
- [License](#license)
- [Authors](#authors)

## Software Components

### Motion Tracking and Swarm Simulation

The Azure Kinect tracker translates performers' physical movements into spatial control information. The swarm simulation introduces collective, autonomous behaviour within a shared virtual environment. These systems extend the possibilities for interaction between human movement, sound, and visual processes.

### Machine Learning and Python Tools

The [CAS_AISoundMotion_2026](https://github.com/bisnad/CAS_AISoundMotion_2026) repository contains Python-based tools for audio analysis, clustering, motion processing, and generative AI. They complement Sonosphere both as standalone tools for preparing material and as interactive components that can be connected to the Sonosphere environment. Please refer to that repository for its current tool list, documentation, dependencies, and installation instructions.

## Installation

- **Clone Sonosphere**

  ```bash
  git clone https://github.com/ewgnr/Sonosphere.git
  cd Sonosphere
  ```

- **Spherical Soundfile Player**

  The C++-based player uses:
  - [openFrameworks](https://openframeworks.cc/)
  - [libsndfile](https://www.mega-nerd.com/libsndfile/)

  Choose the appropriate macOS or Windows version and configure its dependencies.

- **Azure Kinect motion tracking**

  Compatible Azure Kinect hardware and the corresponding SDKs are required, depending on the tracker implementation:
  - Azure Kinect Sensor SDK
  - Azure Kinect Body Tracking SDK

  Check operating-system and hardware compatibility for the SDK versions in use.

- **Machine learning and Python tools (optional)**

  To use the Python tools independently or connect them to Sonosphere, clone their separate repository:

  ```bash
  git clone https://github.com/bisnad/CAS_AISoundMotion_2026.git
  ```

  Follow the installation and environment setup instructions in that repository. Python dependencies are maintained there rather than duplicated in this README.

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- **Share:** Copy and redistribute the material in any medium or format.
- **Adapt:** Remix, transform, and build upon the material.

**Under the following terms:**
- **Attribution:** Give appropriate credit, provide a link to the license, and indicate if changes were made. Do not suggest that the licensor endorses you or your use.
- **NonCommercial:** You may not use the material for commercial purposes.
- **No additional restrictions:** You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

**Notices:** No warranties are given. Other rights, such as publicity, privacy, or moral rights, may limit how you use the material.

## Authors

- [Daniel Bisig](https://github.com/bisnad)
- [Ephraim Wegner](https://github.com/ewgnr)
