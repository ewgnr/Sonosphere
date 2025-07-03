#include "ofApp.h"

void ofApp::setup() {
  
    sender.setup("127.0.0.1", 9004);  // OSC Adress, OSC Port

    ofSetFrameRate(60);
    ofBackground(0);

    try {
        device = k4a::device::open(0);
    }
    catch (...) {
        ofLogError() << "Kinect konnte nicht geöffnet werden.";
        return;
    }

    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;                     // Add all azure Kinect settings below -> https://microsoft.github.io/Azure-Kinect-Sensor-SDK/release/1.4.x/structk4a__record__configuration__t.html
    config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    config.camera_fps = K4A_FRAMES_PER_SECOND_15;

    device.start_cameras(&config);
    calibration = device.get_calibration(config.depth_mode, K4A_COLOR_RESOLUTION_OFF);
    tracker = k4abt::tracker::create(calibration, K4ABT_TRACKER_CONFIG_DEFAULT);

    ofLogNotice() << "Tracker gestartet (Standardmodus, keine CUDA-Unterstützung)";
}

//--------------------------------------------------------------
void ofApp::update() {
    if (!device.get_capture(&capture, std::chrono::milliseconds(0))) {
        return; 
    }

    tracker.enqueue_capture(capture);

    k4abt::frame bodyFrame;
    if (tracker.pop_result(&bodyFrame, std::chrono::milliseconds(0))) {
        size_t numBodies = bodyFrame.get_num_bodies();
        currentBodies.clear();

        for (size_t i = 0; i < numBodies; ++i) {
            k4abt_body_t body = bodyFrame.get_body(i);
            currentBodies.push_back(body);
        }

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
               // pos.z *= -1;  // rotate Z-Axis

                msg.addFloatArg(pos.x);
                msg.addFloatArg(pos.y);
                msg.addFloatArg(pos.z);
            }

            sender.sendMessage(msg);
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw() {
    ofSetColor(255);
    ofDrawBitmapString("Bodies detected: " + ofToString(currentBodies.size()), 20, 20);

    for (const auto& body : currentBodies) {
        for (int i = 0; i < static_cast<int>(K4ABT_JOINT_COUNT); ++i) {
            const auto& joint = body.skeleton.joints[i];

            float x = joint.position.v[0] * 0.005f;
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
        scaleFactor *= 1.1f;
        ofLogNotice() << "ScaleFactor erhöht: " << scaleFactor;
    }
    else if (key == '-' ) {
        scaleFactor *= 0.9f;
        ofLogNotice() << "ScaleFactor verringert: " << scaleFactor;
    }
}



//--------------------------------------------------------------
void ofApp::exit() {
    tracker.shutdown();
    device.close();
}
