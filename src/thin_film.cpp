#include "thin_film.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace {
    struct EdgeKey {
        unsigned int a;
        unsigned int b;

        EdgeKey(unsigned int x, unsigned int y)
        {
            if (x < y) {
                a = x;
                b = y;
            }
            else {
                a = y;
                b = x;
            }
        }

        bool operator==(const EdgeKey& other) const
        {
            return a == other.a && b == other.b;
        }
    };

    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& edge) const
        {
            return static_cast<size_t>(edge.a) * 73856093u ^ static_cast<size_t>(edge.b) * 19349663u;
        }
    };

    float clampFloat(float value, float minValue, float maxValue)
    {
        return std::max(minValue, std::min(maxValue, value));
    }

}



ThinFilmSimulator::ThinFilmSimulator()
    : m_mesh(NULL),
      m_c2(0.0f),
      m_damping(0.0f),
      m_minThickness(0.012f),
      m_maxThickness(0.095f),
      m_initialTotalThickness(0.0f)
{
}

void ThinFilmSimulator::initialize(Mesh* mesh, float initialThickness)
{
    m_mesh = mesh;
    if (!m_mesh) return;

    const size_t count = m_mesh->vertices.size();
    m_h.assign(count, initialThickness);
    m_vh.assign(count, 0.0f);
    m_vertexArea.assign(count, 1.0f);

    buildAdjacency();
    computeVertexAreas();

    for (size_t i = 0; i < count; ++i) {
        const glm::vec3 p = m_mesh->vertices[i].Position;
        const float seed =
            std::sin(9.0f * p.x + 3.0f * p.y) +
            0.6f * std::sin(11.0f * p.z - 4.0f * p.x) +
            0.35f * std::sin(17.0f * (p.x + p.y + p.z));
        m_h[i] = clampFloat(initialThickness + 0.025f * seed, m_minThickness, m_maxThickness);
        m_vh[i] = 0.025f * std::sin(13.0f * p.y + 5.0f * p.z);
    }

    m_initialTotalThickness = 0.0f;
    for (size_t i = 0; i < m_h.size(); ++i) {
        m_initialTotalThickness += m_h[i] * m_vertexArea[i];
    }

    applyToMesh();
}

void ThinFilmSimulator::setThicknessRange(float minThickness, float maxThickness)
{
    m_minThickness = std::min(minThickness, maxThickness);
    m_maxThickness = std::max(minThickness, maxThickness);
}

void ThinFilmSimulator::buildAdjacency()
{
    m_neighbors.assign(m_mesh->vertices.size(), std::vector<unsigned int>());
    std::unordered_set<EdgeKey, EdgeKeyHash> edges;

    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        const unsigned int i0 = m_mesh->indices[i];
        const unsigned int i1 = m_mesh->indices[i + 1];
        const unsigned int i2 = m_mesh->indices[i + 2];

        edges.insert(EdgeKey(i0, i1));
        edges.insert(EdgeKey(i1, i2));
        edges.insert(EdgeKey(i2, i0));
    }

    for (std::unordered_set<EdgeKey, EdgeKeyHash>::const_iterator it = edges.begin(); it != edges.end(); ++it) {
        m_neighbors[it->a].push_back(it->b);
        m_neighbors[it->b].push_back(it->a);
    }
}

void ThinFilmSimulator::computeVertexAreas()
{
    std::fill(m_vertexArea.begin(), m_vertexArea.end(), 0.0f);

    for (size_t i = 0; i + 2 < m_mesh->indices.size(); i += 3) {
        const unsigned int i0 = m_mesh->indices[i];
        const unsigned int i1 = m_mesh->indices[i + 1];
        const unsigned int i2 = m_mesh->indices[i + 2];

        const glm::vec3 p0 = m_mesh->vertices[i0].Position;
        const glm::vec3 p1 = m_mesh->vertices[i1].Position;
        const glm::vec3 p2 = m_mesh->vertices[i2].Position;
        const float area = 0.5f * glm::length(glm::cross(p1 - p0, p2 - p0));
        m_vertexArea[i0] += area / 3.0f;
        m_vertexArea[i1] += area / 3.0f;
        m_vertexArea[i2] += area / 3.0f;
    }

    for (size_t i = 0; i < m_vertexArea.size(); ++i) {
        if (m_vertexArea[i] <= 1e-8f) {
            m_vertexArea[i] = 1.0f;
        }
    }
}

void ThinFilmSimulator::step(float dt)
{
    if (!m_mesh || m_h.empty() || dt <= 0.0f) return;

    std::vector<float> laplacian(m_h.size(), 0.0f);
    for (size_t i = 0; i < m_h.size(); ++i) {
        if (m_neighbors[i].empty()) continue;

        float sum = 0.0f;
        float weightSum = 0.0f;
        for (size_t n = 0; n < m_neighbors[i].size(); ++n) {
            const unsigned int j = m_neighbors[i][n];
            const float weight = 1.0f / std::max(1e-4f, glm::length(m_mesh->vertices[j].Position - m_mesh->vertices[i].Position));
            sum += weight * (m_h[j] - m_h[i]);
            weightSum += weight;
        }

        if (weightSum > 1e-6f) {
            laplacian[i] = sum / weightSum;
        }
    }

    for (size_t i = 0; i < m_h.size(); ++i) {
        m_vh[i] += m_c2 * laplacian[i] * dt;
        m_vh[i] -= m_damping * m_vh[i] * dt;
        m_h[i] = clampFloat(m_h[i] + m_vh[i] * dt, m_minThickness, m_maxThickness);
    }

    conserveGlobalThickness();
    applyToMesh();
}

void ThinFilmSimulator::injectDisturbance(unsigned int centerVertex, float deltaThickness, float radius)
{
    if (!m_mesh || centerVertex >= m_h.size()) return;

    const glm::vec3 center = m_mesh->vertices[centerVertex].Position;
    const float safeRadius = std::max(radius, 1e-4f);
    for (size_t i = 0; i < m_h.size(); ++i) {
        const float dist = glm::length(m_mesh->vertices[i].Position - center);
        if (dist > safeRadius) continue;

        const float falloff = 0.5f + 0.5f * std::cos(3.14159265f * dist / safeRadius);
        m_h[i] = clampFloat(m_h[i] + deltaThickness * falloff, m_minThickness, m_maxThickness);
        m_vh[i] += deltaThickness * falloff * 12.0f;
    }

    conserveGlobalThickness();
    applyToMesh();
}

void ThinFilmSimulator::conserveGlobalThickness()
{
    float currentTotal = 0.0f;
    float activeArea = 0.0f;
    for (size_t i = 0; i < m_h.size(); ++i) {
        currentTotal += m_h[i] * m_vertexArea[i];
        if (m_h[i] > m_minThickness + 1e-5f && m_h[i] < m_maxThickness - 1e-5f) {
            activeArea += m_vertexArea[i];
        }
    }

    if (activeArea <= 1e-8f) return;

    const float correction = (m_initialTotalThickness - currentTotal) / activeArea;
    for (size_t i = 0; i < m_h.size(); ++i) {
        if (m_h[i] <= m_minThickness + 1e-5f || m_h[i] >= m_maxThickness - 1e-5f) continue;
        m_h[i] = clampFloat(m_h[i] + correction, m_minThickness, m_maxThickness);
    }
}

glm::vec3 ThinFilmSimulator::falseColor(float h) const
{
    const float t = clampFloat((h - m_minThickness) / (m_maxThickness - m_minThickness), 0.0f, 1.0f);
    const float phase = t * 6.2831853f;
    return glm::vec3(
        0.5f + 0.5f * std::sin(phase),
        0.5f + 0.5f * std::sin(phase + 2.0943951f),
        0.5f + 0.5f * std::sin(phase + 4.1887902f)
    );
}

void ThinFilmSimulator::applyToMesh()
{
    if (!m_mesh) return;

    const size_t count = std::min(m_h.size(), m_mesh->vertices.size());
    for (size_t i = 0; i < count; ++i) {
        // 물리적 두께만 셰이더로 전달 (Fragment Shader에서 LUT 연산에 사용)
        m_mesh->vertices[i].Thickness = m_h[i]; 
        
        // 가짜 색상 덮어쓰기 방지 (흰색으로 고정하거나 셰이더에서 무시하도록 처리)
        m_mesh->vertices[i].Color = glm::vec3(1.0f); 
    }
}

