#include "pbd.h"
#include <unordered_set>
#include <algorithm>

struct EdgeKey {
    unsigned int a, b;
    EdgeKey(unsigned int x, unsigned int y) {
        if (x < y) { a = x; b = y; }
        else { a = y; b = x; }
    }
    bool operator==(EdgeKey const& o) const { return a == o.a && b == o.b; }
};
namespace std {
    template<> struct hash<EdgeKey> {
        size_t operator()(EdgeKey const& k) const noexcept {
            return (size_t)k.a * 1000003u + k.b;
        }
    };
}

PBDSolver::PBDSolver(Mesh* mesh) : m_mesh(mesh), m_initialVolume(0.0f) {}

void PBDSolver::initialize() {
    m_particles.clear();
    m_edges.clear();
    if (!m_mesh) return;

    m_particles.resize(m_mesh->vertices.size());
    for (size_t i = 0; i < m_mesh->vertices.size(); ++i) {
        auto& v = m_mesh->vertices[i];
        PBDParticle p;
        p.position = v.Position;
        p.prevPosition = v.Position;
        p.acceleration = glm::vec3(0.0f);
        p.invMass = 1.0f;

        // 얇은 막 간섭을 위한 노이즈 섞인 초기 두께
        float seed = std::sin(9.0f * p.position.x + 3.0f * p.position.y) +
            0.6f * std::sin(11.0f * p.position.z - 4.0f * p.position.x);
        p.thickness = 0.05f + 0.01f * seed;
        p.prevThickness = p.thickness;
        p.thicknessVelocity = 0.0f;
        m_particles[i] = p;
    }

    buildConstraints();
    m_initialVolume = computeMeshVolume();
}

void PBDSolver::buildConstraints() {
    std::unordered_set<EdgeKey> uniqueEdges;
    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        uniqueEdges.insert(EdgeKey(m_mesh->indices[i], m_mesh->indices[i + 1]));
        uniqueEdges.insert(EdgeKey(m_mesh->indices[i + 1], m_mesh->indices[i + 2]));
        uniqueEdges.insert(EdgeKey(m_mesh->indices[i + 2], m_mesh->indices[i]));
    }

    for (const auto& edge : uniqueEdges) {
        PBDEdge e;
        e.a = edge.a;
        e.b = edge.b;
        e.restLength = glm::length(m_particles[e.a].position - m_particles[e.b].position);
        m_edges.push_back(e);
    }
}

float PBDSolver::computeMeshVolume() {
    float volume = 0.0f;
    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        glm::vec3 p1 = m_particles[m_mesh->indices[i]].position;
        glm::vec3 p2 = m_particles[m_mesh->indices[i + 1]].position;
        glm::vec3 p3 = m_particles[m_mesh->indices[i + 2]].position;
        volume += glm::dot(p1, glm::cross(p2, p3));
    }
    return volume / 6.0f;
}

void PBDSolver::integrate(float dt, float damping) {
    if (dt <= 0.0f) return;
    float dt2 = dt * dt;
    for (auto& p : m_particles) {
        if (p.invMass == 0.0f) continue;
        glm::vec3 velocity = (p.position - p.prevPosition) * (1.0f - damping);
        glm::vec3 newPos = p.position + velocity + p.acceleration * dt2;
        p.prevPosition = p.position;
        p.position = newPos;
        p.acceleration = glm::vec3(0.0f); // 명시적 힘(Force) 초기화
    }
}



void PBDSolver::solveConstraints(int iterations) {
    const float volumeCompliance = 1e-10f;

    // Jacobi 방식은 연결된 간선 수만큼 나누어지므로(보통 5~6개), 
    // 0.8 정도로 높게 주어야 적당히 출렁이는 유체 막 느낌이 납니다.
    const float edgeStiffness = 0.8f;

    for (int it = 0; it < iterations; ++it) {

        // ==========================================================
        // 1. 자코비(Jacobi) 구조 제약 (표면 매끄러움 유지 & 유령 회전력 완벽 방지)
        // ==========================================================
        std::vector<glm::vec3> edgeCorr(m_particles.size(), glm::vec3(0.0f));
        std::vector<int> edgeCounts(m_particles.size(), 0);

        for (const auto& e : m_edges) {
            glm::vec3 pA = m_particles[e.a].position;
            glm::vec3 pB = m_particles[e.b].position;
            glm::vec3 delta = pB - pA;
            float len = glm::length(delta);

            if (len > 1e-6f) {
                float diff = (len - e.restLength) / len;
                glm::vec3 corr = delta * diff * edgeStiffness * 0.5f;

                // 위치를 즉시 바꾸지 않고 보정량만 누적합니다.
                edgeCorr[e.a] += corr;
                edgeCorr[e.b] -= corr;
                edgeCounts[e.a]++;
                edgeCounts[e.b]++;
            }
        }

        // 루프가 끝난 뒤 모든 정점을 '동시에' 업데이트합니다.
        for (size_t i = 0; i < m_particles.size(); ++i) {
            if (edgeCounts[i] > 0) {
                // 간선 개수로 나누어 평균을 내면 시스템이 절대 폭발하지 않습니다.
                m_particles[i].position += edgeCorr[i] / (float)edgeCounts[i];
            }
        }

        // ==========================================================
        // 2. 전역 부피 제약 (공기 압력 유지)
        // ==========================================================
        float currentVolume = computeMeshVolume();
        float C_vol = currentVolume - m_initialVolume;

        std::vector<glm::vec3> gradVol(m_particles.size(), glm::vec3(0.0f));
        for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
            unsigned int idx1 = m_mesh->indices[i];
            unsigned int idx2 = m_mesh->indices[i + 1];
            unsigned int idx3 = m_mesh->indices[i + 2];

            glm::vec3 p1 = m_particles[idx1].position;
            glm::vec3 p2 = m_particles[idx2].position;
            glm::vec3 p3 = m_particles[idx3].position;

            gradVol[idx1] += glm::cross(p2, p3) / 6.0f;
            gradVol[idx2] += glm::cross(p3, p1) / 6.0f;
            gradVol[idx3] += glm::cross(p1, p2) / 6.0f;
        }

        float sumGradVolSq = 0.0f;
        for (size_t i = 0; i < m_particles.size(); ++i) {
            sumGradVolSq += glm::dot(gradVol[i], gradVol[i]);
        }

        if (sumGradVolSq > 1e-6f) {
            float lambdaVol = -C_vol / (sumGradVolSq + volumeCompliance);
            for (size_t i = 0; i < m_particles.size(); ++i) {
                m_particles[i].position += lambdaVol * gradVol[i];
            }
        }
    }
}



void PBDSolver::recomputeNormals() {
    for (auto& v : m_mesh->vertices) v.Normal = glm::vec3(0.0f);
    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        unsigned int i0 = m_mesh->indices[i];
        unsigned int i1 = m_mesh->indices[i + 1];
        unsigned int i2 = m_mesh->indices[i + 2];
        glm::vec3 p0 = m_mesh->vertices[i0].Position;
        glm::vec3 p1 = m_mesh->vertices[i1].Position;
        glm::vec3 p2 = m_mesh->vertices[i2].Position;
        glm::vec3 normal = glm::cross(p1 - p0, p2 - p0);
        if (glm::length(normal) > 1e-6f) {
            normal = glm::normalize(normal);
            m_mesh->vertices[i0].Normal += normal;
            m_mesh->vertices[i1].Normal += normal;
            m_mesh->vertices[i2].Normal += normal;
        }
    }
    for (auto& v : m_mesh->vertices) {
        if (glm::length(v.Normal) > 1e-6f) v.Normal = glm::normalize(v.Normal);
    }
}



void PBDSolver::integrateThickness(float dt) {
    if (dt <= 0.0f) return;

    // 1. Graph Laplacian 계산 (Topological Laplacian)
    // 인접한 정점들 간의 두께 차이를 누적하여 표면 위의 확산/파동 성분을 구합니다.
    std::vector<float> laplacians(m_particles.size(), 0.0f);

    for (const auto& e : m_edges) {
        float hA = m_particles[e.a].thickness;
        float hB = m_particles[e.b].thickness;

        // 질량(액체) 보존 법칙에 따라 서로 두께를 교환
        float diff = hB - hA;
        laplacians[e.a] += diff;
        laplacians[e.b] -= diff;
    }

    // 2. 2차 동역학계 적분 (파동 방정식)
    float c2 = 100.0f;     // 파동 전파 속도 (시각적 일렁임의 속도 결정)
    float k_damp = 0.0f;   // 감쇠 계수 (파동이 서서히 잦아들게 함)

    for (size_t i = 0; i < m_particles.size(); ++i) {
        auto& p = m_particles[i];

        // 가속도 계산: a = c^2 * ∇^2 h
        float accel = c2 * laplacians[i];

        // 속도 적분 및 감쇠 적용
        p.thicknessVelocity += accel * dt;
        p.thicknessVelocity *= (1.0f - k_damp);

        // 위치(두께) 적분
        p.prevThickness = p.thickness;
        p.thickness += p.thicknessVelocity * dt;

        // 3. 렌더링 안정성을 위한 Clamping
        // 광학적 간섭 무늬가 잘 보이는 나노미터(nm) 범위 밖으로 벗어나지 않도록 제한
        // (shader_lighting.fs의 LUT 텍스처 범위와 호환)
        p.thickness = std::max(0.012f, std::min(p.thickness, 0.095f));
    }
}

void PBDSolver::step(float dt, int solverIterations, const glm::vec3& gravity, float damping) {
    for (auto& p : m_particles) p.acceleration += gravity;
    integrate(dt, damping);
    solveConstraints(solverIterations);
    integrateThickness(dt);
    recomputeNormals();
}

void PBDSolver::applyToMesh() {
    if (!m_mesh) return;
    for (size_t i = 0; i < m_particles.size(); ++i) {
        m_mesh->vertices[i].Position = m_particles[i].position;
        m_mesh->vertices[i].Thickness = m_particles[i].thickness; // 순수 두께만 전달
    }
    m_mesh->updateVertexBuffer();
}

void PBDSolver::addImpulse(unsigned int particleIdx, const glm::vec3& velocity) {
    if (particleIdx >= m_particles.size()) return;

    m_particles[particleIdx].prevPosition -= velocity * 0.001f;
    m_particles[particleIdx*3].prevPosition += velocity * 0.001f;

    
}