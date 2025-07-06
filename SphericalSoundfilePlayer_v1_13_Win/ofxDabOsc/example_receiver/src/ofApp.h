#pragma once

#include "ofMain.h"
#include <memory.h>
#include "dab_osc_parser.h"
#include "dab_osc_sender.h"
#include "dab_osc_receiver.h"

class ofApp : public ofBaseApp {

public:
	void setup();
	void update();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);


	std::shared_ptr<dab::OscParser> oscParser;
	dab::OscSender* oscSender;
	dab::OscReceiver* oscReceiver;

	unsigned int mSentMessageCount = 0;
};
