#pragma once

#include "ofMain.h"
#include "ofxOsc.h"

#include <k4a/k4a.hpp>
#include <k4abt.h>     // Für k4abt_body_t und k4abt_skeleton_t
#include <k4abt.hpp>   // Für k4abt::tracker etc.

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void exit();

private:
    k4a::device device;
    k4a::capture capture;
    k4a::calibration calibration;
    k4abt::tracker tracker;

    ofxOscSender sender;

    float scaleFactor = 0.05f;
    void keyPressed(int key);  // Hinzugefügt


    std::vector<k4abt_body_t> currentBodies;  // Richtiger Typ!
};
