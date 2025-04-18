#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    // Initialisiere das Kinect-Objekt
    kinect.setup();

    // Stelle sicher, dass du den Kinect-Sensor mit der richtigen Konfiguration startest
    kinect.start();
}

//--------------------------------------------------------------
void ofApp::update() {
    // Aktualisiere Kinect-Daten
    kinect.update();

    // Lese Körperdaten aus und speichere sie in dem bodies-Vektor
    bodies = kinect.getBodies();
}

//--------------------------------------------------------------
void ofApp::draw() {
    // Zeichne die Positionen der Gelenke, wenn Körper erkannt wurden
    for (auto& body : bodies) {
        for (int i = 0; i < body.joints.size(); i++) {
            auto joint = body.joints[i];
            ofDrawCircle(joint.position.x, joint.position.y, 10);  // Zeichne die Gelenke als Kreise
        }
    }
}
