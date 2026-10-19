#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    sender_1.setup("127.0.0.1", 9004);  // Spherical Soundfile Player
    //sender_2.setup("192.168.0.6", 9003);  // Parvival Ray Marching  
    sender_2.setup("127.0.0.1", 9003);  // Ray Marching

    ofSetFrameRate(60);
    ofBackground(0);

    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    //config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;  // Stable & fast
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;  // Stable & fast
    config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;

    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU;
    tracker_config.sensor_orientation = K4ABT_SENSOR_ORIENTATION_DEFAULT;

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

    tracker = k4abt::tracker::create(calibration, tracker_config);
    ofLogNotice() << "Tracker gestartet (Standardmodus, CUDA aktiviert)";
}

//--------------------------------------------------------------
void ofApp::update() {
    std::vector<k4abt_body_t> tempBodies;

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

    // search for closest body
    k4abt::frame bodyFrame;

    if (tracker.pop_result(&bodyFrame, std::chrono::milliseconds(30)))
    {
        numBodies = bodyFrame.get_num_bodies();
        foundTrackedPerson = false;

        if (numBodies > 0) foundTrackedPerson = true;

        float closestHipDistance = 10000000.0;

        for (size_t i = 0; i < numBodies; ++i)
        {
            k4abt_body_t body = bodyFrame.get_body(i);
            const auto& joints = body.skeleton.joints;
            float hipDistance = joints[K4ABT_JOINT_PELVIS].position.v[2] * scaleFactor;

            if (hipDistance < closestHipDistance)
            {
                closestHipDistance = hipDistance;
                closestBody = body;
            }

            //std::cout << "body " << i << " hipDist " << hipDistance << " closestHipDistance " << closestHipDistance << "\n";
        }

        if (closestHipDistance > minBodyHipDistance)
        {
            numBodies = 0;
            foundTrackedPerson == false;
        }

        //std::cout << "numBodies " << numBodies << "\n";

        // send osc message (numbodies)
        {
            ofxOscMessage msg;
            msg.setAddress("/mocap/numbodies");
            msg.addIntArg(numBodies);
            sender_1.sendMessage(msg);
            sender_2.sendMessage(msg);
        }
        // send osc message (joint pos)
        if (foundTrackedPerson == true)
        {
            ofxOscMessage msg;
            msg.setAddress("/mocap/joint/pos_world");

            const auto& joints = closestBody.skeleton.joints;

            glm::vec3 root = {
                joints[K4ABT_JOINT_PELVIS].position.v[0] * scaleFactor,
                joints[K4ABT_JOINT_PELVIS].position.v[1] * scaleFactor,
                joints[K4ABT_JOINT_PELVIS].position.v[2] * scaleFactor
            };

            //std::cout << "root x " << root[0] << " y " << root[1] << " z " << root[2] << "\n";

            for (int i = 0; i < static_cast<int>(K4ABT_JOINT_COUNT); ++i) {
                glm::vec3 pos = {
                    joints[i].position.v[0] * scaleFactor,
                    joints[i].position.v[1] * scaleFactor,
                    joints[i].position.v[2] * scaleFactor
                };

                pos -= root;
                float ty = pos.y;
                pos.y = -pos.z;
                pos.z = ty;

                pos.x *= -1;
                pos.y *= -1;
                pos.z *= -1;

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
    ofDrawBitmapString("Bodies detected: " + ofToString(drawBodies.size()), 20, 20);
    ofDrawBitmapString("Mirror Mode: " + string(mirrorMode ? "ON" : "OFF"), 20, 40);

    if (numBodies > 0)
    {
        for (int i = 0; i < static_cast<int>(K4ABT_JOINT_COUNT); ++i) {
            const auto& joint = closestBody.skeleton.joints[i];

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
void ofApp::keyPressed(int key) {

    if (key == '+') {
        minBodyHipDistance += minBodyHipDistanceIncr;

        std::cout << "minBodyHipDistance erhöht: " << minBodyHipDistance << "\n";
        ofLogNotice() << "minBodyHipDistance erhöht: " << minBodyHipDistance;
    }
    else if (key == '-') {
        minBodyHipDistance -= minBodyHipDistanceIncr;

        std::cout << "minBodyHipDistance erhöht: " << minBodyHipDistance << "\n";
        ofLogNotice() << "minBodyHipDistance verringert: " << minBodyHipDistance;
    }
    
    //if (key == '+') {
    //    scaleFactor *= 1.1f;
    //    ofLogNotice() << "ScaleFactor erhöht: " << scaleFactor;
    //}
    //else if (key == '-') {
    //    scaleFactor *= 0.9f;
    //    ofLogNotice() << "ScaleFactor verringert: " << scaleFactor;
    //}
    else if (key == 'm' || key == 'M') {
        mirrorMode = !mirrorMode;
        ofLogNotice() << "Mirror Mode: " << (mirrorMode ? "ON" : "OFF");
    }
    else if (key == 'l' || key == 'L') {
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
