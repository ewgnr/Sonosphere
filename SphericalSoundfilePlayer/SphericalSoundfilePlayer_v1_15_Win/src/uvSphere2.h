#pragma once

#include "ofMain.h"
#include "ofxDatGui.h"
#include "vertexMapping.h"
#include <map>
#include <vector>
#include <cmath>

class uvSphere
{
public:
    uvSphere(int stacks = 20, int slices = 20, float r = 1.0f)
        : radius(r), numStacks(stacks), numSlices(slices)
    {
        buildVertices();

        int vertex_count = uniqueVertices.size();
        offsets = std::vector<float>(vertex_count, r * 100.0);
    }

    void buildVertices()
    {
        uniqueVertices.clear();
        vertexToIndex.clear();
        vertexIndex = 0;

        for (int i = 0; i <= numStacks; ++i)
        {
            float lat = PI * float(i) / float(numStacks);
            float sinLat = sin(lat);
            float cosLat = cos(lat);

            for (int j = 0; j <= numSlices; ++j)
            {
                float lon = TWO_PI * float(j) / float(numSlices);
                float sinLon = sin(lon);
                float cosLon = cos(lon);

                glm::vec3 v;
                v.x = sinLat * cosLon;
                v.y = cosLat;
                v.z = sinLat * sinLon;
                v *= radius;

                glm::vec3 rounded = roundVec(v, 5);

                if (vertexToIndex.find(rounded) == vertexToIndex.end())
                {
                    vertexToIndex[rounded] = vertexIndex++;
                    uniqueVertices.push_back(v);
                }
            }
        }
    }

    void buildMeshWithInputs(const std::vector<VertexMapping>& vertexMappings)
    {
        of_mesh.clear();
        of_mesh.setMode(OF_PRIMITIVE_POINTS);

        for (std::size_t idx = 0; idx < uniqueVertices.size(); ++idx)
        {
            float multiplier = 1.0f;

            auto it = std::find_if(vertexMappings.begin(), vertexMappings.end(),
                [idx](const VertexMapping& vm) {
                    return vm.vertexIndex == idx;
                });

            if (it != vertexMappings.end() && it->guiInput)
            {
                multiplier = ofToFloat(it->guiInput->getText());
            }

            glm::vec3 adjusted = glm::normalize(uniqueVertices[idx]) * multiplier;
            of_mesh.addVertex(adjusted);
        }
    }

    const std::vector<glm::vec3>& getUniqueVertices() const { return uniqueVertices; }
    ofMesh& getMesh() { return of_mesh; }
    const std::vector<float>& getOffsets() const { return offsets; }

    void setOffset(int index, float offset) { offsets[index] = offset; }

private:
    float radius = 1.0f;
    int numStacks = 20;
    int numSlices = 20;

    struct Vec3Compare {
        bool operator()(const glm::vec3& a, const glm::vec3& b) const {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        }
    };

    glm::vec3 roundVec(const glm::vec3& v, int decimals) const {
        float scale = pow(10.0f, decimals);
        return glm::vec3(
            roundf(v.x * scale) / scale,
            roundf(v.y * scale) / scale,
            roundf(v.z * scale) / scale
        );
    }

    std::vector<glm::vec3> uniqueVertices;
    std::map<glm::vec3, int, Vec3Compare> vertexToIndex;
    ofMesh of_mesh;
    int vertexIndex = 0;
    std::vector<float> offsets;
};
