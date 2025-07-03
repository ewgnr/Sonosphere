#define _HAS_STD_BYTE 0

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    // Kinect initialisieren
    kinect.open();
    kinect.initBodySource();

    // OSC Sender einrichten
    sender.setup("127.0.0.1", 9004); // IP und Port für SphericalSoundfilePlayer
}

//--------------------------------------------------------------
void ofApp::update() {
    kinect.update();
}

//--------------------------------------------------------------
void ofApp::draw() {
    ofBackground(0);
    ofDrawBitmapString("Kinect Skeleton OSC Sender", 20, 20);

    if (kinect.isFrameNew()) {
        auto bodies = kinect.getBodySource()->getBodies();
        for (auto& body : bodies) {
            if (body.tracked) {
                ofxOscMessage msg;
               msg.setAddress("/mocap/joint/pos_world"); // Erwartete OSC Adresse 

                float scale = 100.0f;
                // 25 Body Points über OSC
                glm::vec3 root;
                std::array<glm::vec3, 25> _joints;

                auto joint = body.joints[(JointType)0];
                root = { joint.getPosition().x * scale, joint.getPosition().y * scale, joint.getPosition().z * scale };

                for (int i = 0; i < 25; i++)
                {
                    auto joint = body.joints[(JointType)i];

                    _joints[i] = { joint.getPosition().x * scale, joint.getPosition().y * scale, joint.getPosition().z * scale };

                    _joints[i] -= root;

                    // ab hier unten verschicken
                }
                for (int i = 0; i < 25; i++) {
                    auto joint = _joints[i];
                    msg.addFloatArg(joint.x);
                    msg.addFloatArg(joint.y);
                    msg.addFloatArg(joint.z);

                    // Punkte für Zeichnung mittig Zentrieren
                     // Skalierungsfaktor für Sichtbarkeit
                    float x = ofGetWidth() / 2 + joint.x;
                    float y = ofGetHeight() / 2 - joint.y;

                    // Punkte zeichnen
                    ofSetColor(0, 255, 0); // Grün
                    ofDrawCircle(x, y, 5);

                    // Gelenk-ID anzeigen
                    ofSetColor(255);
                    ofDrawBitmapString(ofToString(i), x + 10, y);
                }


                sender.sendMessage(msg, false);
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::exit() {
    kinect.close();
}
