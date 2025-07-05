#pragma once

#include "ofMain.h"

class uvSphere
{
public:
	uvSphere() : n_stacks(0), n_slices(0), countStacks(0), radius(0) {}

	void setup(const size_t& pStacks, const size_t& pSlices, const double pRadius)
	{
		n_stacks = pStacks;
		n_slices = pSlices;
		radius = pRadius;  
	}

	void verticies()
	{
		// add top vertex
		vert_pnt = { 0, 1, 0 };
		ref_vert_pnt.push_back(vert_pnt * radius);
		indiciesString.push_back(std::to_string(0));

		// add vertices per stack / slice
		for (size_t i = 1; i < n_stacks; i++)
		{
			double phi = PI * double(i) / double(n_stacks);

			for (size_t j = 0; j < n_slices; j++)
			{
				double theta = 2.0 * PI * double(j) / double(n_slices);

				vert_pnt.x = std::sin(phi) * std::cos(theta);
				vert_pnt.y = std::cos(phi);
				vert_pnt.z = std::sin(phi) * std::sin(theta);

				ref_vert_pnt.push_back(vert_pnt * radius);
				indiciesString.push_back(std::to_string((i - 1) * n_stacks + j));
			}
		}

		// add bottom vertex
		vert_pnt = { 0, -1, 0 };
		ref_vert_pnt.push_back(vert_pnt * radius);
		indiciesString.push_back(std::to_string((n_stacks - 2) * n_stacks + n_slices));
	}

	void mesh()
	{
		// add top / bottom triangles
		for (size_t i = 0; i < n_slices; ++i)
		{
			size_t i0 = i + 1;
			size_t i1 = (i + 1) % n_slices + 1;

			tri_pnt_1.push_back(ref_vert_pnt[0]);
			tri_pnt_2.push_back(ref_vert_pnt[i1]);
			tri_pnt_3.push_back(ref_vert_pnt[i0]);

			i0 = i + n_slices * (n_stacks - 2) + 1;
			i1 = (i + 1) % n_slices + n_slices * (n_stacks - 2) + 1;

			tri_pnt_1.push_back(ref_vert_pnt[ref_vert_pnt.size() - 1]);
			tri_pnt_2.push_back(ref_vert_pnt[i1]);
			tri_pnt_3.push_back(ref_vert_pnt[i0]);
		}

		// add quads per stack / slice
		for (size_t i = 0; i < n_stacks - 2; ++i)
		{
			size_t i0 = i * n_slices + 1;
			size_t i1 = (i + 1) * n_slices + 1;

			for (size_t j = 0; j < n_slices; ++j)
			{
				rect_pnt_1.push_back(ref_vert_pnt[i0 + j]);
				rect_pnt_2.push_back(ref_vert_pnt[i0 + (j + 1) % n_slices]);
				rect_pnt_3.push_back(ref_vert_pnt[i1 + (j + 1) % n_slices]);
			}
		}

		for (size_t i = 0; i < rect_pnt_1.size(); ++i)
		{
			line.begin();
			line.addVertex(rect_pnt_1[i]);
			line.addVertex(rect_pnt_2[i]);
			line.addVertex(rect_pnt_3[i]);
			line.end();
		}
	}

	void drawVerticies(const ofColor& pColor)
	{
		ofPushStyle();
		ofSetColor(pColor);
		sphere.setRadius(1);
		for (auto const& v : ref_vert_pnt)
		{
			sphere.setPosition(v);
			sphere.draw();
		}
		ofPopStyle();
	}

	void drawMesh(const ofColor& pColor)
	{
		ofPushStyle();
		ofSetColor(pColor);
		ofNoFill();
		for (size_t i = 0; i < tri_pnt_1.size(); ++i)
		{
			ofTriangle(tri_pnt_1[i], tri_pnt_2[i], tri_pnt_3[i]);
		}

		line.draw();
		ofPopStyle();
	}

	void drawIndicies(const ofColor& pColor)
	{
		ofPushStyle();
		ofSetColor(pColor);
		for (size_t i = 0; i < ref_vert_pnt.size(); ++i)
		{
			ofPushMatrix();
			ofTranslate(ref_vert_pnt[i]);
			ofDrawBitmapString(indiciesString[i], 0, 10);
			ofPopMatrix();
		}
		ofPopStyle();
	}

	void drawExitingPosition(const size_t pIndex, const ofColor& pColor)
	{
		ofPushStyle();
		ofSetColor(pColor);
		sphere.setRadius(2);
		sphere.setPosition(ref_vert_pnt[pIndex]);
		sphere.draw();
		ofPopStyle();
	}

	double radius;
	std::vector<glm::vec3> ref_vert_pnt;
	std::vector<std::string> indiciesString;

private:

	size_t n_stacks, n_slices;
	int countStacks;
	glm::vec3 vert_pnt;
	std::vector<glm::vec3> tri_pnt_1, tri_pnt_2, tri_pnt_3;
	std::vector<glm::vec3> rect_pnt_1, rect_pnt_2, rect_pnt_3, rect_pnt_4;

	ofSpherePrimitive sphere;
	ofPolyline line;
};