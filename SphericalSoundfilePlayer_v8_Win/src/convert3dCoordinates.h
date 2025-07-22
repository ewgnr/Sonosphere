#pragma once

#include "ofMain.h"

class convert3dCoordinates
{
public:
	convert3dCoordinates() {}

	inline double distance(const glm::vec3& pV1, const glm::vec3& pV2)
	{
		return std::sqrt(std::pow(pV2.x - pV1.x, 2) + std::pow(pV2.y - pV1.y, 2) + std::pow(pV2.z - pV1.z, 2));
	}
};