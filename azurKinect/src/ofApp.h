#pragma once

#include "ofMain.h"
#include "ofxAzureKinect.h"  // Importiere die Azure Kinect Bibliothek

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();

    ofxAzureKinect kinect;  // Deklaration der Azure Kinect Instanz

    vector<ofxAzureKinect::Body> bodies; // Vektor für die Körperdaten
};
