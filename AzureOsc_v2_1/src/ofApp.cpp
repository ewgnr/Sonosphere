#include "ofApp.h"

void ofApp::setup() {

    sender_1.setup("127.0.0.1", 9004);  // Spherical Soundfile Player
    sender_2.setup("127.0.0.1", 9003);  // Parvival Ray Marching  

    // Add all azure Kinect settings below -> https://microsoft.github.io/Azure-Kinect-Sensor-SDK/release/1.4.x/structk4a__record__configuration__t.html
    ofSetFrameRate(60);
    ofBackground(0);

    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;


    //Switch between live tracking and recording (L)
    if (useLiveTracking) {
        try {
            device = k4a::device::open(0);
            device.start_cameras(&config);
            calibration = device.get_calibration(config.depth_mode, K4A_COLOR_RESOLUTION_OFF);
            ofLogNotice() << "Live-Kinect erfolgreich gestartet";
        }
        catch (...) {
            ofLogError() << "Kinect konnte nicht geöffnet werden.";
            useLiveTracking = false; 
        }
    }

    if (!useLiveTracking) {
        if (k4a_playback_open("recording.mkv", &player) != K4A_RESULT_SUCCEEDED) {
            ofLogError() << "Failed to open recording";
        }
        else {
            k4a_playback_get_calibration(player, &calibration);
            ofLogNotice() << "Recording erfolgreich geladen";
        }
    }

    tracker = k4abt::tracker::create(calibration, K4ABT_TRACKER_CONFIG_DEFAULT);
    ofLogNotice() << "Tracker gestartet (Standardmodus, keine CUDA-Unterstützung)";
}

//--------------------------------------------------------------
void ofApp::update() {
    currentBodies.clear();

    if (useLiveTracking) {
        k4a::capture liveCapture;
        if (device.get_capture(&liveCapture, std::chrono::milliseconds(10))) {
            tracker.enqueue_capture(liveCapture);
        }
        else {
            return;
        }
    }
    else {
        k4a_stream_result_t result = k4a_playback_get_next_capture(player, &capture);
        if (result == K4A_STREAM_RESULT_SUCCEEDED) {
            tracker.enqueue_capture(capture);
        }
        else {
            k4a_playback_seek_timestamp(player, 0, K4A_PLAYBACK_SEEK_BEGIN);
            return;
        }
    }

    k4abt::frame bodyFrame;
    if (tracker.pop_result(&bodyFrame, std::chrono::milliseconds(10))) {
        size_t numBodies = bodyFrame.get_num_bodies();

        bool foundTrackedPerson = false;
        for (size_t i = 0; i < numBodies; ++i) {
            k4abt_body_t body = bodyFrame.get_body(i);
            if (body.id == trackedBodyId) {
                currentBodies.push_back(body);
                foundTrackedPerson = true;
                break;
            }
        }

        // Wenn noch keine Person getrackt wird, erste speichern
        if (!foundTrackedPerson && numBodies > 0) {
            k4abt_body_t firstBody = bodyFrame.get_body(0);
            trackedBodyId = firstBody.id;
            currentBodies.push_back(firstBody);
            ofLogNotice() << "Neue Person getrackt: Body ID = " << trackedBodyId;
        }

        // Wenn keine Person erkannt → Tracking-ID zurücksetzen
        if (numBodies == 0) {
            trackedBodyId = std::numeric_limits<uint32_t>::max();
        }

        // OSC-Daten senden
        for (const auto& body : currentBodies) {
            ofxOscMessage msg;
            msg.setAddress("/mocap/joint/pos_world");

            const auto& joints = body.skeleton.joints;

            glm::vec3 root = {
                joints[K4ABT_JOINT_PELVIS].position.v[0] * scaleFactor,
                joints[K4ABT_JOINT_PELVIS].position.v[1] * scaleFactor,
                joints[K4ABT_JOINT_PELVIS].position.v[2] * scaleFactor
            };

            for (int i = 0; i < static_cast<int>(K4ABT_JOINT_COUNT); ++i) {
                glm::vec3 pos = {
                    joints[i].position.v[0] * scaleFactor,
                    joints[i].position.v[1] * scaleFactor,
                    joints[i].position.v[2] * scaleFactor
                };

                pos -= root;
                pos.y *= -1;
                if (mirrorMode) pos.x *= -1;

                msg.addFloatArg(pos.x);
                msg.addFloatArg(pos.y);
                msg.addFloatArg(pos.z);
            }

            sender_1.sendMessage(msg);
            sender_2.sendMessage(msg);
        }
    }
}




//--------------------------------------------------------------
void ofApp::draw() {
    ofSetColor(255);
    ofDrawBitmapString("Bodies detected: " + ofToString(currentBodies.size()), 20, 20);
    ofDrawBitmapString("Mirror Mode: " + string(mirrorMode ? "ON" : "OFF"), 20, 40);


    for (const auto& body : currentBodies) {
        for (int i = 0; i < static_cast<int>(K4ABT_JOINT_COUNT); ++i) {
            const auto& joint = body.skeleton.joints[i];

            float x = joint.position.v[0] * 0.005f;
            if (mirrorMode) x *= -1;

            float y = joint.position.v[1] * 0.005f;

            float centerX = ofGetWidth() / 2;
            float centerY = ofGetHeight() / 2;

            ofDrawCircle(x * 50 + centerX, y * 50 + centerY, 5);
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {                   // Scaling of skeleton 
    if (key == '+') {
        scaleFactor *= 1.1f;
        ofLogNotice() << "ScaleFactor erhöht: " << scaleFactor;
    }
    else if (key == '-') {
        scaleFactor *= 0.9f;
        ofLogNotice() << "ScaleFactor verringert: " << scaleFactor;
    }

    else if (key == 'm' || key == 'M') {              //mirror on x axis
    mirrorMode = !mirrorMode;
    ofLogNotice() << "Mirror Mode: " << (mirrorMode ? "ON" : "OFF");
    }

    else if (key == 'l' || key == 'L') {                // toggle between live tracking and recording
        useLiveTracking = !useLiveTracking;
        tracker.shutdown();
        tracker = nullptr;
        device.close();
        k4a_playback_close(player);

        setup(); 
        ofLogNotice() << "Modus gewechselt zu: " << (useLiveTracking ? "LIVE" : "RECORDING");
    }
}



//--------------------------------------------------------------
void ofApp::exit() {
    tracker.shutdown();
    if (useLiveTracking) {
        device.close();
    }
    else {
        k4a_playback_close(player);
    }
}
