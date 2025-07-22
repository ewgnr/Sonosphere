#pragma once

#include <memory>
#include <glm/glm.hpp>
#include "ofxDatGui.h"   // for ofxDatGuiTextInput*
#include "sfPlayer.h"    // for std::shared_ptr<sfPlayer>

struct VertexMapping
{
    std::size_t guiIndex;
    std::size_t vertexIndex;
    glm::vec3 position;
    std::shared_ptr<sfPlayer> player;
    ofxDatGuiTextInput* guiInput = nullptr;
};
