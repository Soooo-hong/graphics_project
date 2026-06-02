#include "pbd.h"
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>

PBDSolver::PBDSolver(Mesh* mesh)
    : m_mesh(mesh)
{
}

void PBDSolver::initialize()
{
    m_particles.clear();
    m_constraints.clear();

    if (!m_mesh) return;

    m_particles.resize(m_mesh->vertices.size());
    for (size_t i = 0; i < m_mesh->vertices.size(); ++i) {
        const auto& v = m_mesh->vertices[i];
        PBDParticle p;
        p.position = v.Position;
        p.prevPosition = v.Position;
        p.acceleration = glm::vec3(0.0f);
        p.invMass = 1.0f;
        m_particles[i] = p;
    }

    m_neighbors.assign(m_particles.size(), std::vector<unsigned int>());
    float initialThickness = 0.05f;

    for (size_t i = 0; i < m_particles.size(); ++i) {
        glm::vec3 p = m_particles[i].position;

        // 공간 좌표를 기반으로 사인파 노이즈를 섞어 두께를 다르게 만듭니다.
        float seed = std::sin(9.0f * p.x + 3.0f * p.y) +
            0.6f * std::sin(11.0f * p.z - 4.0f * p.x) +
            0.35f * std::sin(17.0f * (p.x + p.y + p.z));

        // 기본 두께 0.05에 노이즈를 더해 0.04 ~ 0.06 사이의 굴곡진 두께 형성
        float noiseThickness = 0.05f + 0.01f * seed;

        m_particles[i].thickness = noiseThickness;
        m_particles[i].prevThickness = noiseThickness;
        m_particles[i].thicknessVelocity = 0.0f;
        m_particles[i].baseArea = 0.0f;
    }

    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        unsigned int i0 = m_mesh->indices[i];
        unsigned int i1 = m_mesh->indices[i + 1];
        unsigned int i2 = m_mesh->indices[i + 2];

        glm::vec3 p0 = m_mesh->vertices[i0].Position;
        glm::vec3 p1 = m_mesh->vertices[i1].Position;
        glm::vec3 p2 = m_mesh->vertices[i2].Position;

        // 삼각형의 면적 계산 후 각 정점에 1/3씩 분배
        float triArea = 0.5f * glm::length(glm::cross(p1 - p0, p2 - p0));
        m_particles[i0].baseArea += triArea / 3.0f;
        m_particles[i1].baseArea += triArea / 3.0f;
        m_particles[i2].baseArea += triArea / 3.0f;
    }

    m_initialTotalVolume = 0.0f;
    for (size_t i = 0; i < m_particles.size(); ++i) {
        m_initialTotalVolume += initialThickness * m_particles[i].baseArea;
    }

    buildConstraintsFromTriangles();

    computeCotangentWeights();
}

struct EdgeKey {
    unsigned int a, b;
    EdgeKey(unsigned int x, unsigned int y) {
        if (x < y) { a = x; b = y; }
        else { a = y; b = x; }
    }
    bool operator==(EdgeKey const& o) const { return a == o.a && b == o.b; }
};
struct EdgeHash {
    size_t operator()(EdgeKey const& k) const noexcept {
        return (size_t)k.a * 1000003u + k.b;
    }
};

void PBDSolver::computeCotangentWeights()
{
    m_cotWeights.assign(m_particles.size(), std::vector<CotWeight>());
    std::vector<std::unordered_map<unsigned int, float>> weightMatrix(m_particles.size());

    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        unsigned int i0 = m_mesh->indices[i];
        unsigned int i1 = m_mesh->indices[i + 1];
        unsigned int i2 = m_mesh->indices[i + 2];

        glm::vec3 p0 = m_mesh->vertices[i0].Position;
        glm::vec3 p1 = m_mesh->vertices[i1].Position;
        glm::vec3 p2 = m_mesh->vertices[i2].Position;

        // 코탄젠트 계산 람다 함수 (dot(a,b) / length(cross(a,b)))
        auto cotangent = [](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
            glm::vec3 ba = a - b;
            glm::vec3 bc = c - b;
            float cosT = glm::dot(ba, bc);
            float sinT = glm::length(glm::cross(ba, bc));
            return cosT / (sinT + 1e-6f);
            };

        float cot0 = cotangent(p1, p0, p2);
        float cot1 = cotangent(p2, p1, p0);
        float cot2 = cotangent(p0, p2, p1);

        weightMatrix[i0][i1] += 0.5f * cot2; weightMatrix[i1][i0] += 0.5f * cot2;
        weightMatrix[i1][i2] += 0.5f * cot0; weightMatrix[i2][i1] += 0.5f * cot0;
        weightMatrix[i2][i0] += 0.5f * cot1; weightMatrix[i0][i2] += 0.5f * cot1;
    }

    for (size_t i = 0; i < weightMatrix.size(); ++i) {
        for (auto const& pair : weightMatrix[i]) {
            float clampedW = std::max(pair.second, 0.0f); // 물리적 불안정(둔각) 방지
            m_cotWeights[i].push_back({ pair.first, clampedW });
        }
    }
}

void PBDSolver::buildConstraintsFromTriangles()
{
    if (!m_mesh) return;

    std::unordered_set<EdgeKey, EdgeHash> edges;
    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        unsigned int i0 = m_mesh->indices[i];
        unsigned int i1 = m_mesh->indices[i + 1];
        unsigned int i2 = m_mesh->indices[i + 2];

        EdgeKey e0(i0, i1), e1(i1, i2), e2(i2, i0);
        edges.insert(e0); edges.insert(e1); edges.insert(e2);
    }

    m_constraints.reserve(edges.size());
    for (auto const& e : edges) {
        PBDConstraint c;
        c.a = e.a;
        c.b = e.b;
        glm::vec3 pa = m_particles[c.a].position;
        glm::vec3 pb = m_particles[c.b].position;
        c.restLength = glm::length(pb - pa);
        m_constraints.push_back(c);

        m_neighbors[e.a].push_back(e.b);
        m_neighbors[e.b].push_back(e.a);
    }

    for (size_t i = 0; i < m_particles.size(); ++i) {
        for (size_t j = i + 1; j < m_particles.size(); ++j) {
            if (glm::length(m_particles[i].position - m_particles[j].position) < 1e-4f) {
                PBDConstraint weld;
                weld.a = i;
                weld.b = j;
                weld.restLength = 0.0f;
                m_constraints.push_back(weld);
            }
        }
    }
}

void PBDSolver::integrate(float dt, float damping)
{
    if (dt <= 0.0f) return;
    float dt2 = dt * dt;
    for (auto& p : m_particles) {
        if (p.invMass == 0.0f) {
            p.acceleration = glm::vec3(0.0f);
            continue;
        }
        glm::vec3 velocity = (p.position - p.prevPosition) * (1.0f - damping);
        glm::vec3 newPos = p.position + velocity + p.acceleration * dt2;
        p.prevPosition = p.position;
        p.position = newPos;
        p.acceleration = glm::vec3(0.0f);
    }
}

void PBDSolver::solveConstraints(int iterations)
{
    if (iterations <= 0) iterations = 1;

    float stiffness = 0.2f;

    for (int it = 0; it < iterations; ++it) {
        for (auto& c : m_constraints) {
            PBDParticle& A = m_particles[c.a];
            PBDParticle& B = m_particles[c.b];
            glm::vec3 delta = B.position - A.position;
            float len = glm::length(delta);
            if (len <= 1e-6f) continue;
            float diff = (len - c.restLength) / len;

            float w1 = A.invMass;
            float w2 = B.invMass;
            float wsum = w1 + w2;
            if (wsum == 0.0f) continue;

            glm::vec3 correction = delta * diff * stiffness;
            if (A.invMass > 0.0f) A.position += correction * (w1 / wsum);
            if (B.invMass > 0.0f) B.position -= correction * (w2 / wsum);
        }
    }
}

void PBDSolver::recomputeNormals()
{
    for (auto& v : m_mesh->vertices) {
        v.Normal = glm::vec3(0.0f);
    }

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
        else v.Normal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
}


void PBDSolver::step(float dt, int solverIterations, const glm::vec3& gravity, float damping)
{
    recomputeNormals();

    float internalPressure = 2.0f;     // 비눗방울 팽창력 (내부 기압)
    float surfaceTension = 10.0f;      // 표면 장력 (곡률 복원력)

    for (size_t i = 0; i < m_particles.size(); ++i) {
        if (m_particles[i].invMass > 0.0f) {
            m_particles[i].acceleration += gravity;

            // 1. 내부 압력 팽창 (법선 방향)
            m_particles[i].acceleration += m_mesh->vertices[i].Normal * internalPressure;

            // 2. 표면 장력 (평균 곡률 H 기반 복원력)
            glm::vec3 curvatureForce(0.0f);
            float weightSum = 0.0f;
            for (auto& cw : m_cotWeights[i]) {
                curvatureForce += cw.w * (m_particles[cw.j].position - m_particles[i].position);
                weightSum += cw.w;
            }
            if (weightSum > 1e-6f) {
                curvatureForce /= weightSum;
            }
            // 곡률 벡터(curvatureForce)는 표면적을 줄이려는 방향(중심)을 향함
            m_particles[i].acceleration += surfaceTension * curvatureForce;
        }
    }

    // 1. Move particles based on velocity and acceleration
    integrate(dt, damping);

    // 2. Enforce structural integrity (distance constraints)
    solveConstraints(solverIterations);

    // 3. (Optional) Update thickness for the interference colors
    integrateThickness(dt);



}

void PBDSolver::applyToMesh()
{
    if (!m_mesh) return;
    size_t n = std::min(m_particles.size(), m_mesh->vertices.size());

    for (size_t i = 0; i < n; ++i) {
        // 1. 위치 업데이트
        m_mesh->vertices[i].Position = m_particles[i].position;

        // 2. 가짜 RGB 계산 로직 제거, 순수 두께 값만 Vertex Attribute로 전달
        m_mesh->vertices[i].Thickness = m_particles[i].thickness;

        // (선택) 디버깅용이 아니라면 Color는 기본값(흰색)으로 두거나 셰이더에서 무시하면 됩니다.
        // m_mesh->vertices[i].Color = glm::vec3(1.0f); 
    }

    recomputeNormals();
    m_mesh->updateVertexBuffer();
}

void PBDSolver::addImpulse(unsigned int particleIdx, const glm::vec3& velocity)
{
    if (particleIdx >= m_particles.size() || m_particles[particleIdx].invMass == 0.0f)
        return;

    // 위치 충격
    m_particles[particleIdx].acceleration += velocity * 500.0f;
    m_particles[particleIdx].thicknessVelocity += 0.05f;
}

void PBDSolver::integrateThickness(float dt)
{
    // 1. 파동 전파 속도 대폭 증가 (이웃에게 힘을 전달할 수 있도록 20000.0f로 설정)
    float c2 = 200.0f;
    float k_damp = 0.0f;  // 파동이 너무 영원히 지속되지 않도록 약간의 감쇠

    // 2. In-place 업데이트 방지: 모든 파티클의 Laplacian(가속도)을 먼저 계산해서 임시 저장
    std::vector<float> laplacians(m_particles.size(), 0.0f);

    for (size_t i = 0; i < m_particles.size(); ++i) {
        float laplacian = 0.0f;
        float weightSum = 0.0f;

        // 코탄젠트 가중치를 이용한 이산 라플라스-벨트라미 연산
        for (auto& cw : m_cotWeights[i]) {
            // 현재 프레임의 오리지널 thickness 끼리만 비교하도록 보장
            laplacian += cw.w * (m_particles[cw.j].thickness - m_particles[i].thickness);
            weightSum += cw.w;
        }

        if (weightSum > 1e-6f) {
            laplacian /= weightSum;
        }
        laplacians[i] = laplacian;
    }

    // 3. 계산된 Laplacian을 바탕으로 모든 파티클을 일괄 업데이트
    for (size_t i = 0; i < m_particles.size(); ++i) {
        auto& p = m_particles[i];

        float thicknessAccel = c2 * laplacians[i];
        p.thicknessVelocity += thicknessAccel * dt;
        p.thicknessVelocity *= (1.0f - k_damp);

        p.prevThickness = p.thickness;
        p.thickness += p.thicknessVelocity * dt;

        // 두께가 너무 얇아져서 위상차가 깨지는 현상 방지
        if (p.thickness < 0.001f) p.thickness = 0.001f;
    }

    for (int iter = 0; iter < 2; ++iter) {
        for (auto& c : m_constraints) {
            // 거리가 0인 제약 = 완전히 동일한 위치에 겹쳐있는 분리된 정점(Weld)
            if (c.restLength == 0.0f) {
                auto& A = m_particles[c.a];
                auto& B = m_particles[c.b];

                // 두 정점의 두께와 파동 속도를 평균내서 똑같이 맞춰줌
                float avgThickness = (A.thickness + B.thickness) * 0.5f;
                float avgVel = (A.thicknessVelocity + B.thicknessVelocity) * 0.5f;

                A.thickness = avgThickness;
                B.thickness = avgThickness;
                A.thicknessVelocity = avgVel;
                B.thicknessVelocity = avgVel;
            }
        }
    }

}

void PBDSolver::solveThicknessConstraints(const std::vector<float>& currentAreas)
{
    float currentTotalVolume = 0.0f;
    float totalArea = 0.0f;

    for (size_t i = 0; i < m_particles.size(); ++i) {
        currentTotalVolume += m_particles[i].thickness * currentAreas[i];
        totalArea += currentAreas[i];
    }

    float C = currentTotalVolume - m_initialTotalVolume;

    if (totalArea > 1e-6f) {
        // 면적 팽창에 따른 피드백 루프를 끊고 일괄적으로 보정
        float deltaH = -C / totalArea;

        for (size_t i = 0; i < m_particles.size(); ++i) {
            m_particles[i].thickness += deltaH;
            if (m_particles[i].thickness < 0.001f) m_particles[i].thickness = 0.001f;
        }
    }
}