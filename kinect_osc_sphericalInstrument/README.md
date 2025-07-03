- Echtzeit-Erkennung von Skeleton-Daten (Kinect v2)
- Sendet 3D-Koordinaten der 25 Gelenkpunkte über OSC
- Visualisiert Gelenkpunkte in der App

## 🛠️ Voraussetzungen

- Windows 10 oder 11
- KinectSDK https://www.microsoft.com/en-us/download/details.aspx?id=44561
- OpenFrameworks 0.11.2 oder neuer
- KinectforWindows2 Openframeworks Addon: https://github.com/elliotwoods/ofxKinectForWindows2?utm_source=chatgpt.com
- Visual Studio 2019 oder 2022
- Kinect v2 Sensor mit Netzteil und USB 3.0
  

## 📦 Installation

1. Windows kinect2 SDK installieren 
2. KinectforWindows2 addon zu Ofx Addons hinzufügen
3. update  Project mit  Project Generator (Achtung auf addons: ofxosc + kinectforwindows2)
4. in VS -> Projekteinstellungen 
   - c++ > Addons > zusätzliche Include Verzeichnisse > C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\inc ( installationspfad des Kinect SDK)
   
   - Linker > Allgeimein > C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\Lib\x64\ (installationspfad zu den .lib des KinectSDK)
   - Linker > Eingabe >  Kinect20.lib 
   




## 🎚️ OSC Einstellungen

- **IP:** `127.0.0.1`
- **Port:** `9004`
- **OSC-Adresse:** `/mocap/joint/pos_world`
- **Pro Nachricht:** 25 × 3 (x, y, z) Float-Werte
