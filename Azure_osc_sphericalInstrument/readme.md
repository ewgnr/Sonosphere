ofxAzureKinect Skeleton OSC Sender
Dieses Projekt verwendet das ofxAzureKinect Addon für OpenFrameworks, um in Echtzeit Skelettdaten vom Azure Kinect Sensor zu erfassen.
Die 3D-Positionen der 32 Gelenkpunkte werden per OSC versendet und können z.B. für Motion Capture oder Visualisierungen genutzt werden.

Echtzeit-Skelettverfolgung mit Azure Kinect
Senden von 3D-Koordinaten (x, y, z) der Gelenkpunkte über OSC
Live-Visualisierung der Gelenkpunkte in der App
Dynamische Skalierung der Ausgabe über Tastatur (+ / -)

Installation & Setup
Install AzureSDK and Addon:
SDK: https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/sensor-sdk-download
Addon: https://github.com/prisonerjohn/ofxAzureKinect.git
In VS Code add Linker to SDK Files. 
Copy files from SDK Bin Folder in Project Bin Folder!

OSC Einstellungen
IP-Adresse: 127.0.0.1
Port: 9004
OSC-Adresse: /mocap/joint/pos_world
Nachrichtenformat: 32 Gelenke × 3 Float-Werte (x, y, z) pro Gelenk

Bedienung
Mit der Taste + / = den Skalierungsfaktor erhöhen
Mit der Taste - / _ den Skalierungsfaktor verringern


