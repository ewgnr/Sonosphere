#pragma once
#include "ofMain.h"
#include "ofxDatGui.h"
#include "vertexMapping.h"
#include <array>
#include <map>
#include <vector>

class icoSphere
{
public:
    using Triangle = std::array<glm::vec3, 3>;

    icoSphere(int subdivisionLevel = 0, float r = 1.0f)
        : radius(r), currentSubdivisionLevel(subdivisionLevel)
    {
        rebuild(subdivisionLevel);
    }

    void rebuild(int subdivisionLevel)
    {
        currentSubdivisionLevel = subdivisionLevel;
        base = generateIcosahedron();
        highRes = (subdivisionLevel > 0) ? subdivide(base, subdivisionLevel) : base;

        vertexIndex = 0;
        vertexToIndex.clear();
        uniqueVertices.clear();

        for (const auto& tri : highRes)
        {
            for (const auto& v : tri)
            {
                if (vertexToIndex.find(v) == vertexToIndex.end())
                {
                    vertexToIndex[v] = vertexIndex++;
                    uniqueVertices.push_back(v);
                }
            }
        }
    }

    const std::vector<glm::vec3>& getUniqueVertices() const {
        return uniqueVertices;
    }

    void setVertexInputs(const std::vector<ofxDatGuiTextInput*>& inputs)
    {
        vertexInputs = inputs;
    }

    void buildMesh(ofPrimitiveMode mode = OF_PRIMITIVE_POINTS)
    {
        if (vertexInputs.empty() || vertexToIndex.empty()) return;

        of_mesh.clear();
        of_mesh.setMode(mode);

        for (const auto& tri : highRes)
        {
            for (const auto& original : tri)
            {
                glm::vec3 normalized = glm::normalize(original);
                int idx = vertexToIndex[original];
                float multiplier = ofToFloat(vertexInputs[idx]->getText());
                glm::vec3 adjusted = normalized * multiplier;
                of_mesh.addVertex(adjusted);
            }
        }
    }

    void buildMeshWithInputs(const std::vector<VertexMapping>& vertexMappings, ofPrimitiveMode mode = OF_PRIMITIVE_POINTS)
    {
        if (vertexMappings.empty() || vertexToIndex.empty()) return;

        of_mesh.clear();
        of_mesh.setMode(mode);

        for (const auto& tri : highRes)
        {
            for (const auto& original : tri)
            {
                auto itIndex = vertexToIndex.find(original);
                if (itIndex == vertexToIndex.end()) continue;
                int vertexIdx = itIndex->second;

                auto it = std::find_if(vertexMappings.begin(), vertexMappings.end(),
                    [vertexIdx](const VertexMapping& vm) {
                        return vm.vertexIndex == vertexIdx;
                    });

                float multiplier = 1.0f;
                if (it != vertexMappings.end() && it->guiInput)
                {
                    multiplier = ofToFloat(it->guiInput->getText());
                }

                glm::vec3 normalized = glm::normalize(original);
                glm::vec3 adjusted = normalized * multiplier;
                of_mesh.addVertex(adjusted);
            }
        }
    }

    ofMesh& getMesh()
    {
        return of_mesh;
    }

    const std::vector<Triangle>& getBase() const { return base; }
    const std::vector<Triangle>& getHighRes() const { return highRes; }

    float getRadius() const { return radius; }

    void setRadius(float r)
    {
        radius = r;
        rebuild(currentSubdivisionLevel);
    }

private:
    Triangle make_triangle(const glm::vec3& pV0, const glm::vec3& pV1, const glm::vec3& pV2) const
    {
        return { pV0, pV1, pV2 };
    }

    std::vector<Triangle> generateIcosahedron() const
    {
        float phi = (1.0f + sqrt(5.0f)) * 0.5f;
        float a = 1.0f * radius;
        float b = (1.0f / phi) * radius;

        std::vector<glm::vec3> vertices =
        {
            {0, b, -a}, {b, a, 0}, {-b, a, 0}, {0, b, a}, {0, -b, a}, {-a, 0, b},
            {0, -b, -a}, {a, 0, -b}, {a, 0, b}, {-a, 0, -b}, {b, -a, 0}, {-b, -a, 0}
        };

        return
        {
            make_triangle(vertices[2], vertices[1], vertices[0]),
            make_triangle(vertices[1], vertices[2], vertices[3]),
            make_triangle(vertices[5], vertices[4], vertices[3]),
            make_triangle(vertices[4], vertices[8], vertices[3]),
            make_triangle(vertices[7], vertices[6], vertices[0]),
            make_triangle(vertices[6], vertices[9], vertices[0]),
            make_triangle(vertices[11], vertices[10], vertices[4]),
            make_triangle(vertices[10], vertices[11], vertices[6]),
            make_triangle(vertices[9], vertices[5], vertices[2]),
            make_triangle(vertices[5], vertices[9], vertices[11]),
            make_triangle(vertices[8], vertices[7], vertices[1]),
            make_triangle(vertices[7], vertices[8], vertices[10]),
            make_triangle(vertices[2], vertices[5], vertices[3]),
            make_triangle(vertices[8], vertices[1], vertices[3]),
            make_triangle(vertices[9], vertices[2], vertices[0]),
            make_triangle(vertices[1], vertices[7], vertices[0]),
            make_triangle(vertices[11], vertices[9], vertices[6]),
            make_triangle(vertices[7], vertices[10], vertices[6]),
            make_triangle(vertices[5], vertices[11], vertices[4]),
            make_triangle(vertices[10], vertices[8], vertices[4])
        };
    }

    std::vector<Triangle> subdivide(const std::vector<Triangle>& input, int level) const
    {
        if (cachedSubdivisions.count(level)) return cachedSubdivisions.at(level);

        std::map<std::pair<glm::vec3, glm::vec3>, glm::vec3, Vec3PairCompare> midpointCache;
        std::vector<Triangle> result = input;

        for (int i = 0; i < level; ++i)
        {
            std::vector<Triangle> newResult;

            for (const auto& tri : result)
            {
                const glm::vec3& v1 = tri[0];
                const glm::vec3& v2 = tri[1];
                const glm::vec3& v3 = tri[2];

                auto get_midpoint = [&](const glm::vec3& a, const glm::vec3& b)
                {
                    auto key = make_edge_key(a, b);
                    auto it = midpointCache.find(key);
                    if (it != midpointCache.end()) return it->second;

                    glm::vec3 midpoint = glm::normalize(a + b) * radius;
                    midpointCache[key] = midpoint;
                    return midpoint;
                };

                glm::vec3 a = get_midpoint(v1, v2);
                glm::vec3 b = get_midpoint(v2, v3);
                glm::vec3 c = get_midpoint(v3, v1);

                newResult.push_back({ v1, a, c });
                newResult.push_back({ a, v2, b });
                newResult.push_back({ c, b, v3 });
                newResult.push_back({ a, b, c });
            }

            result = std::move(newResult);
        }

        cachedSubdivisions[level] = result;
        return result;
    }

    struct Vec3Compare 
    {
        bool operator()(const glm::vec3& a, const glm::vec3& b) const 
        {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        }
    };

    struct Vec3PairCompare 
    {
        bool operator()(const std::pair<glm::vec3, glm::vec3>& a,
            const std::pair<glm::vec3, glm::vec3>& b) const {
            Vec3Compare cmp;
            if (cmp(a.first, b.first)) return true;
            if (cmp(b.first, a.first)) return false;
            return cmp(a.second, b.second);
        }
    };

    std::pair<glm::vec3, glm::vec3> make_edge_key(const glm::vec3& a, const glm::vec3& b) const 
    {
        Vec3Compare cmp;
        return cmp(a, b) ? std::make_pair(a, b) : std::make_pair(b, a);
    }

    float radius = 1.0f;
    int currentSubdivisionLevel = 0;

    std::vector<Triangle> base;
    std::vector<Triangle> highRes;
    std::map<glm::vec3, int, Vec3Compare> vertexToIndex;
    std::vector<glm::vec3> uniqueVertices;
    std::map<int, std::vector<Triangle>> mutable cachedSubdivisions;

    std::vector<ofxDatGuiTextInput*> vertexInputs;
    ofMesh of_mesh;
    int vertexIndex = 0;
};
