#pragma once
#include "mesh.h"
#include <vector>
#include <glm/glm.hpp>

struct PBDParticle {
    glm::vec3 position;
    glm::vec3 prevPosition;
    glm::vec3 acceleration;
    float invMass;
    float thickness;
    float prevThickness;
    float thicknessVelocity;
};

struct PBDEdge {
    unsigned int a, b;
    float restLength;
};

class PBDSolver {
public:
    PBDSolver(Mesh* mesh);
    void initialize();
    void step(float dt, int solverIterations, const glm::vec3& gravity = glm::vec3(0.0f), float damping = 0.015f);
    void applyToMesh();
    void addImpulse(unsigned int particleIdx, const glm::vec3& velocity);

private:
    Mesh* m_mesh;
    std::vector<PBDParticle> m_particles;
    std::vector<PBDEdge> m_edges;

    float m_initialVolume;

    void buildConstraints();
    void integrate(float dt, float damping);
    void solveConstraints(int iterations);
    void recomputeNormals();
    void integrateThickness(float dt);

    float computeMeshVolume();
};