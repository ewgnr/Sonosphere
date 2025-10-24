#pragma once

#include "ofMain.h"
#include "Flock.h"
#include "Boid.h"
#include "ofxDatGui.h"
#include "ofxOsc.h"

class ofApp : public ofBaseApp
{
	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
		Flock flock;
		ofEasyCam cam;
		ofLight pointLight;

		ofSpherePrimitive sphere;
		std::vector<glm::vec3> attractorPoints;
		glm::vec3 movingTarget;

		void onSliderEvent(ofxDatGuiSliderEvent e);

		ofxDatGui* flocking_GUI = nullptr;
		float separationDistance = 50.0f;
		float separationWeight = 2.0f;
		float alignmentDistance = 80.0f;
		float alignmentWeight = 1.0f;
		float cohesionDistance = 200.0f;
		float cohesionWeight = 1.5f;
		float maxSpeed = 3.5f;
		float maxForce = 0.2f;
		float targetSeekStrength = 0.1f;
		float sphereSeekStrength = 0.12f;
		float radius = 320.0f;
		float theta = 90.0f;
		float phi = 0.0f;

		std::vector<std::string> flocking_GUI_FaderNames =
		{
			"SEPARATION_DIST", "SEPARATION_WEIGHT", "ALIGMENT_DIST", 
			"ALIGMENT_WEIGHT", "COHENSION_DIST", "COHENSION_WEIGHT", 
			"MAX_SPEED", "MAX_FORCE", "TARGET_SEEK_STRENGTH", "SPHERE_SEEK_STRENGTH",
			"TARGET_RADIUS", "TARGET_THETA", "TARGET_PHI",
			"TARGET_RADIUS", "TARGET_THETA", "TARGET_PHI"
		};

		ofxDatGui* presetControl_GUI = nullptr;
		bool mGuiUnlockPresets = false;
		bool mGuiReadOrWritePreset = false;
		std::string mGuiSelectPresetNumber;

		void onToggleEvent(ofxDatGuiToggleEvent e);
		void onTextInputEvent(ofxDatGuiTextInputEvent e);
		void saveGuiValuesAsFile(const std::string& presetNumber);
		void onButtonEvent(ofxDatGuiButtonEvent e);
		void readGuiValuesFromFile(const std::string& presetNumber);

		ofFbo boidFbo;
		bool drawDebugElements = false;

		ofxOscSender sender;
		std::string host = "127.0.0.1";
		int port = 9004;
};
