#define _HAS_STD_BYTE 0

#pragma once


#include "ofMain.h"
#include "ofxKinectForWindows2.h"
#include "ofxOsc.h"

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void exit();

private:
    ofxKFW2::Device kinect;
    ofxOscSender sender;
};
