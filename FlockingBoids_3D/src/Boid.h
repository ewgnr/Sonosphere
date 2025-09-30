#pragma once

#include "ofMain.h"

class Boid
{
public:
    Boid(float tempX, float tempY, float tempZ) :
        r(2.0f), maxspeed(3.5f), maxforce(0.2f), position(glm::vec3(tempX, tempY, tempZ)), 
        acceleration(glm::vec3(0, 0, 0))
    {
        float theta = ofRandom(TWO_PI);
        float phi = ofRandom(PI);
        velocity = glm::vec3(
            std::cos(theta) * std::sin(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(phi)
        );
    }

    void run(const std::vector<Boid>& pBoids,
        const std::vector<glm::vec3>* attractor,
        const glm::vec3* target,
        float separationD,
        float separationW,
        float alignmentD,
        float alignmentW,
        float cohesionD,
        float cohesionW,
        float targetStrength,
        float sphereStrength)
    {
        flock(pBoids, attractor, separationD, separationW, alignmentD, alignmentW, cohesionD, cohesionW, sphereStrength);
        
        if (target) 
        {
            steerTowardTarget(*target, targetStrength);
        }
        
        update();
        borders();
    }

    void setDynamics(float speed, float force) {
        maxspeed = speed;
        maxforce = force;
    }

    void applyForce(const glm::vec3& pForce)
    {
        acceleration += pForce / mass;
    }

    void flock(const std::vector<Boid>& pBoids, 
        const std::vector<glm::vec3>* attractors,
        float separationD,
        float separationW,
        float alignmentD,
        float alignmentW,
        float cohesionD,
        float cohesionW,
        float sphereStrength)
    {
        glm::vec3 sep = separate(pBoids, separationD);
        glm::vec3 ali = align(pBoids, alignmentD);
        glm::vec3 coh = cohesion(pBoids, cohesionD);
        sep *= separationW;
        ali *= alignmentW;
        coh *= cohesionW;
        applyForce(sep);
        applyForce(ali);
        applyForce(coh);

        if (attractors && !attractors->empty())
        {
            glm::vec3 totalAttraction(0);
            for (const glm::vec3& point : *attractors)
            {
                float distance = glm::distance(position, point);
                if (distance < 300)
                {
                    glm::vec3 dir = glm::normalize(point);
                    float latitude = std::acos(dir.y);
                    float areaWeight = std::sin(latitude);

                    float falloff = 1.0f - (distance / 300.0f);
                    falloff = glm::clamp(falloff, 0.0f, 1.0f);

                    float strength = sphereStrength * areaWeight * falloff;

                    if (glm::length(point) < 0.001f)
                    {
                        strength *= 0.0f;
                    }

                    totalAttraction += attract(point, strength);
                }
            }

            applyForce(totalAttraction);
        }
    }

    void update()
    {
        velocity += acceleration;
        velocity *= damping;

        if (glm::length2(velocity) > maxspeed * maxspeed)
        {
            velocity = glm::normalize(velocity) * maxspeed;
        }

        position += velocity;
        acceleration = glm::vec3(0, 0, 0);

        trail.push_back(position);
        if (trail.size() > trailLength)
        {
            trail.pop_front();
        }
    }

    glm::vec3 seek(const glm::vec3& pTarget)
    {
        glm::vec3 desired = pTarget - position;
        desired = glm::normalize(desired) * maxspeed;

        glm::vec3 steer = desired - velocity;

        if (glm::length2(steer) > maxforce * maxforce)
        {
            steer = glm::normalize(steer) * maxforce;
        }

        return steer;
    }

    void display() const
    {
        for (size_t i = 1; i < trail.size(); ++i)
        {
            float alpha = ofMap(i, 1, trail.size(), 20, 200);
            ofSetColor(180, alpha);
            ofDrawLine(trail[i - 1], trail[i]);
        }

        ofPushMatrix();
        ofTranslate(position);

        glm::vec3 dir = glm::normalize(velocity);
        glm::vec3 up(0, 1, 0);
        glm::vec3 axis = glm::cross(up, dir);
        float dotProd = glm::clamp(glm::dot(up, dir), -1.0f, 1.0f);
        float angle = acos(dotProd);

        ofRotateDeg(glm::degrees(angle), axis.x, axis.y, axis.z);

        ofSetColor(200);
        ofDrawCone(glm::vec3(0, 0, 0), 3, 10);
        ofPopMatrix();
    }

    void borders()
    {
        float maxRadius = 600;

        float distFromCenter = glm::length(position);
        if (distFromCenter > maxRadius) 
        {
            position = glm::normalize(position) * maxRadius;
            velocity = glm::reflect(velocity, glm::normalize(position));
        }
    }

    void steerTowardTarget(const glm::vec3& target, float strength)
    {
        glm::vec3 steer = seek(target) * strength;
        applyForce(steer);
    }

    glm::vec3 attract(const glm::vec3& target, float strength = 1.0f)
    {
        glm::vec3 desired = target - position;
        float distance = glm::length(desired);
        if (distance > 0)
        {
            desired = glm::normalize(desired) * maxspeed;

            glm::vec3 steer = desired - velocity;
            if (glm::length(steer) > maxforce)
            {
                steer = glm::normalize(steer) * maxforce;
            }

            return steer * strength;
        }
        return glm::vec3(0);
    }

    glm::vec3 separate(const std::vector<Boid>& boids, float seperateD)
    {
        float desiredSeparation = seperateD;
        float desiredSeparation2 = desiredSeparation * desiredSeparation;

        glm::vec3 steer(0);
        int count = 0;

        for (const Boid& other : boids)
        {
            float d2 = glm::distance2(position, other.position);
            if ((d2 > 0) && (d2 < desiredSeparation2)) {
                float d = std::sqrt(d2);
                glm::vec3 diff = position - other.position;
                diff = glm::normalize(diff);
                diff /= d;
                steer += diff;
                count++;
            }
        }

        if (count > 0)
        {
            steer /= static_cast<float>(count);
        }

        if (glm::length2(steer) > 0.0f)
        {
            steer = glm::normalize(steer) * maxspeed - velocity;

            if (glm::length2(steer) > maxforce * maxforce)
            {
                steer = glm::normalize(steer) * maxforce;
            }
        }
        return steer;
    }

    glm::vec3 align(const std::vector<Boid>& bBoids, float aligmentD)
    {
        float neighborDist = aligmentD;
        float neighborDist2 = neighborDist * neighborDist;

        glm::vec3 sum(0);
        int count = 0;

        for (const Boid& other : bBoids) {
            float d2 = glm::distance2(position, other.position);
            if ((d2 > 0) && (d2 < neighborDist2)) {
                sum += other.velocity;
                count++;
            }
        }

        if (count > 0)
        {
            sum /= static_cast<float>(count);
            sum = glm::normalize(sum) * maxspeed;
            glm::vec3 steer = sum - velocity;

            if (glm::length2(steer) > maxforce * maxforce)
            {
                steer = glm::normalize(steer) * maxforce;
            }

            return steer;
        }

        return glm::vec3(0, 0, 0);
    }

    glm::vec3 cohesion(const std::vector<Boid>& boids, float cohensionD)
    {
        float neighborDist = cohensionD;
        float neighborDist2 = neighborDist * neighborDist;

        glm::vec3 sum(0);
        int count = 0;

        for (const Boid& other : boids)
        {
            float d2 = glm::distance2(position, other.position);
            if ((d2 > 0) && (d2 < neighborDist2)) {
                sum += other.position;
                count++;
            }
        }

        if (count > 0)
        {
            sum /= static_cast<float>(count);
            return seek(sum);
        }

        return glm::vec3(0, 0, 0);
    }

    glm::vec3 getPosition() const 
    {
        return position;
    }

private:
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 acceleration;

    float mass = 2.0f;
    float damping = 0.98f;

    float r;
    float maxforce;
    float maxspeed;

    const float worldSize = 800;
    const float depth = 1000;

    std::deque<glm::vec3> trail;
    size_t trailLength = 100;
};
