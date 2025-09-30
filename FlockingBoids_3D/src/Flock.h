#pragma once

#include "ofMain.h"
#include "Boid.h"

class Flock
{
public:

    Flock() {}

    void run(const std::vector<glm::vec3>* attractor, 
        const glm::vec3* target,
        float separationD,
        float separationW,
        float alignmentD,
        float alignmentW,
        float cohesionD,
        float cohesionW,
        float speedW,
        float forceW,
        float targetStrength,
        float sphereStrength)
    {
        for (Boid& b : boids)
        {
            b.setDynamics(speedW, forceW);
            b.run(boids, attractor, 
                target, 
                separationD, 
                separationW, 
                alignmentD, 
                alignmentW, 
                cohesionD, 
                cohesionW, 
                targetStrength, 
                sphereStrength);
        }
    }

    void draw()
    {
        for (const Boid& b : boids)
        {
            b.display();
        }
    }

    void addBoid(const Boid& b)
    {
        boids.push_back(b);
    }

    const std::vector<Boid>& getBoids() const 
    { 
        return boids; 
    }

private:
    std::vector<Boid> boids;
};