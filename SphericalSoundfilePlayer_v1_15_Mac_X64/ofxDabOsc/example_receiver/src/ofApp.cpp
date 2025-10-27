#include "ofApp.h"
#include <sstream>
#include "osc/OscReceivedElements.h"
#include "osc/OscOutboundPacketStream.h"
#include "dab_osc_arg.h"
#include "dab_osc_message.h"
#include "dab_osc_sender.h"
#include "dab_osc_receiver.h"
#include "dab_osc_parser.h"

template<class T>
void print(T var, std::string end = "\n") {
	std::cout << var << " is of type " << typeid(var).name() << end;
}

//--------------------------------------------------------------
void ofApp::setup()
{
	try
	{
		dab::OscUtils& oscUtils = dab::OscUtils::get();

		oscReceiver = new dab::OscReceiver("OscReceiver", 9001);
		oscParser = std::shared_ptr<dab::OscParser>(new dab::OscParser());
		oscReceiver->registerOscListener(std::weak_ptr<dab::OscParser>(oscParser));

		oscReceiver->start();
	}
	catch (dab::Exception& e)
	{
		std::cout << e << "\n";
	}

}

//--------------------------------------------------------------
void ofApp::update()
{
}

//--------------------------------------------------------------
void ofApp::draw() {

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
