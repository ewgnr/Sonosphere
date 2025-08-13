#pragma once

#include "ofMain.h"
#include "ofxOsc.h"

#include <k4a/k4a.hpp>
#include <k4abt.h>     
#include <k4abt.hpp>  
#include <k4arecord/playback.h>

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);
    void exit();

private:

    // Azure 
    k4a::device device;
    k4a_playback_t player;
    k4a_capture_t capture;
    k4a::calibration calibration;
    k4abt::tracker tracker;

    bool useLiveTracking = false;  // default: Live tracking

    //OSC
    ofxOscSender sender_1, sender_2;

    // Skeleton
    float scaleFactor = 0.05f;
    bool mirrorMode = true;
    std::vector<k4abt_body_t> currentBodies;  

    uint32_t trackedBodyId = std::numeric_limits<uint32_t>::max(); 

};
