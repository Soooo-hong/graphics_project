#define GLM_ENABLE_EXPERIMENTAL
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include "opengl_utils.h"
#include "geometry_primitives.h"
#include <iostream>
#include <vector>
#include "camera.h"
#include "texture.h"
#include "texture_cube.h"
#include "model.h"
#include "mesh.h"
#include "scene.h"
#include "math_utils.h"
#include "light.h"
#include "pbd.h"
#include <map>
#include <cmath>

struct Edge {
    unsigned int v1, v2;
    Edge(unsigned int a, unsigned int b) {
        v1 = std::min(a, b);
        v2 = std::max(a, b);
    }
    bool operator<(const Edge& other) const {
        if (v1 != other.v1) return v1 < other.v1;
        return v2 < other.v2;
    }
};

Mesh createIcosphereMesh(unsigned int subdivisions)
{
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // 1. 초기 정이십면체(Icosahedron) 생성
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f; // 황금비

    std::vector<glm::vec3> baseVertices = {
        {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
        { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
        { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
    };

    auto addVertex = [&](glm::vec3 p) -> unsigned int {
        Vertex v;
        v.Position = glm::normalize(p);
        v.Normal = v.Position;

        // 구면 좌표계를 이용한 기본 UV
        float u = 0.5f + std::atan2(v.Position.z, v.Position.x) / (2.0f * glm::pi<float>());
        float v_tex = 0.5f - std::asin(v.Position.y) / glm::pi<float>();
        v.TexCoords = glm::vec2(u, v_tex);

        // 극점 처리(Gimbal lock 방지)를 포함한 탄젠트 벡터 계산
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        if (std::abs(v.Normal.y) > 0.999f) up = glm::vec3(1.0f, 0.0f, 0.0f);
        v.Tangent = glm::normalize(glm::cross(up, v.Normal));

        v.Color = glm::vec3(1.0f);
        v.Thickness = 0.05f; // 초기 두께 세팅
        vertices.push_back(v);
        return vertices.size() - 1;
        };

    for (const auto& p : baseVertices) {
        addVertex(p);
    }

    indices = {
        0, 11, 5,  0, 5, 1,  0, 1, 7,  0, 7, 10,  0, 10, 11,
        1, 5, 9,  5, 11, 4,  11, 10, 2,  10, 7, 6,  7, 1, 8,
        3, 9, 4,  3, 4, 2,  3, 2, 6,  3, 6, 8,  3, 8, 9,
        4, 9, 5,  2, 4, 11,  6, 2, 10,  8, 6, 7,  9, 8, 1
    };

    // 2. 표면 분할(Subdivision) 및 중복 정점 병합(Welding)
    std::map<Edge, unsigned int> midPointCache;

    auto getMidPoint = [&](unsigned int v1, unsigned int v2) -> unsigned int {
        Edge edge(v1, v2);
        // 이미 계산된 중간점이 있으면 캐시에서 반환 (완벽한 구조적 공유)
        if (midPointCache.find(edge) != midPointCache.end()) {
            return midPointCache[edge];
        }
        glm::vec3 mid = (vertices[v1].Position + vertices[v2].Position) * 0.5f;
        unsigned int index = addVertex(mid);
        midPointCache[edge] = index;
        return index;
        };

    for (unsigned int i = 0; i < subdivisions; ++i) {
        std::vector<unsigned int> newIndices;
        midPointCache.clear();

        for (size_t j = 0; j < indices.size(); j += 3) {
            unsigned int v1 = indices[j];
            unsigned int v2 = indices[j + 1];
            unsigned int v3 = indices[j + 2];

            unsigned int a = getMidPoint(v1, v2);
            unsigned int b = getMidPoint(v2, v3);
            unsigned int c = getMidPoint(v3, v1);

            newIndices.push_back(v1); newIndices.push_back(a); newIndices.push_back(c);
            newIndices.push_back(v2); newIndices.push_back(b); newIndices.push_back(a);
            newIndices.push_back(v3); newIndices.push_back(c); newIndices.push_back(b);
            newIndices.push_back(a);  newIndices.push_back(b); newIndices.push_back(c);
        }
        indices = newIndices;
    }

    return Mesh(vertices, indices);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window, DirectionalLight* sun);

bool isWindowed = true;
bool isKeyboardDone[1024] = { 0 };

// setting
const unsigned int SCR_WIDTH = 1600;
const unsigned int SCR_HEIGHT = 1200;
const unsigned int SHADOW_WIDTH = 1024;
const unsigned int SHADOW_HEIGHT = 1024;
const float planeSize = 15.f;

// camera
Camera camera(glm::vec3(-1.0f, 1.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool useNormalMap = true;
bool useSpecular = true;
bool useShadow = false;
bool useLighting = true;

bool usePCF = false;
bool showWireframe = false;

float pbdDamping = 0.00001f;
int pbdSolverIterations = 2;


int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Soap Bubble Simulator", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // build and compile our shader program
    // ------------------------------------
    Shader lightingShader("../shaders/shader_lighting.vs", "../shaders/shader_lighting.fs"); // you can name your shader files however you like
    Shader shadowShader("../shaders/shadow.vs", "../shaders/shadow.fs");
    Shader skyboxShader("../shaders/shader_skybox.vs", "../shaders/shader_skybox.fs");
    Shader wireframeShader("../shaders/wireframe.vs", "../shaders/wireframe.fs");

    Model yourOwnModel;
    yourOwnModel.mesh = createIcosphereMesh(4);
    yourOwnModel.diffuse = nullptr;
    yourOwnModel.normal = nullptr;
    yourOwnModel.specular = nullptr;

    yourOwnModel.VAO = yourOwnModel.mesh.VAO;
    yourOwnModel.mesh.setupMesh(); // 꼭 호출해서 버퍼를 GPU에 묶어주어야 합니다.

    PBDSolver* spherePBD = new PBDSolver(&yourOwnModel.mesh);
    spherePBD->initialize();



    // Add entities to scene.
    // you can change the position/orientation.
    Scene scene;

    // add your model's entity here!
    Entity* sphereEntity = new Entity(&yourOwnModel, glm::vec3(-1, 1, -1), 0.0f, 0.0f, 0.0f, 1.0);
    scene.addEntity(sphereEntity);

    // define depth texture
    DepthMapTexture depth = DepthMapTexture(SHADOW_WIDTH, SHADOW_HEIGHT);


    // skybox
	int skyboxType = 0; // 0: sea, 1: park

    std::vector<std::string> faces
    {
        "../resources/skybox/right.jpg",
        "../resources/skybox/left.jpg",
        "../resources/skybox/top.jpg",
        "../resources/skybox/bottom.jpg",
        "../resources/skybox/front.jpg",
        "../resources/skybox/back.jpg"
    };

    if (skyboxType == 1) {
        std::vector<std::string> faces
        {
            "../resources/sea/right.jpg",
            "../resources/sea/left.jpg",
            "../resources/sea/top.jpg",
            "../resources/sea/bottom.jpg",
            "../resources/sea/front.jpg",
            "../resources/sea/back.jpg"
        };
    }


    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CubemapTexture skyboxTexture = CubemapTexture(faces);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    ThinFilmLUTTexture thinFilmLUT(256, 128);
    unsigned int VAOskybox, VBOskybox;
    getPositionVAO(skybox_positions, sizeof(skybox_positions), VAOskybox, VBOskybox);


    lightingShader.use();
    lightingShader.setInt("material.diffuseSampler", 0);
    lightingShader.setInt("material.specularSampler", 1);
    lightingShader.setInt("material.normalSampler", 2);
    lightingShader.setInt("depthMapSampler", 3);
    lightingShader.setInt("skyboxTexture", 4);
    lightingShader.setInt("thinFilmLUT", 5);
    lightingShader.setFloat("material.shininess", 64.f);    // set shininess to constant value.


    skyboxShader.use();
    skyboxShader.setInt("skyboxTexture1", 0);


    DirectionalLight sun(30.0f, 30.0f, glm::vec3(0.8f));
    sun.updateLightDir();

    // timing setup for fixed timestep
    float oldTime = glfwGetTime();
    float accumulator = 0.0f;
    const float FIXED_DT = 1.0f / 60.0f;
    const float MAX_DT = 0.1f;

    while (!glfwWindowShouldClose(window))
    {
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            spherePBD->addImpulse(10, glm::vec3(5.0f, 2.0f, 0.0f));
        }

        float currentTime = glfwGetTime();
        float dt = currentTime - oldTime;
        oldTime = currentTime;

        if (dt > MAX_DT) {
            dt = MAX_DT;
        }

        deltaTime = dt;
        accumulator += dt;

        processInput(window, &sun);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//glm::vec3 gravity = glm::vec3(0.0f, -9.81f, 0.0f);
        glm::vec3 gravity = glm::vec3(0.0f, 0.0f, 0.0f);
        const int SUB_STEPS = 2;

        while (accumulator >= FIXED_DT) {
            float subDt = FIXED_DT / SUB_STEPS;
            if (spherePBD) {
                for (int i = 0; i < SUB_STEPS; i++) {
                    spherePBD->step(subDt, pbdSolverIterations, gravity, pbdDamping);
                }
            }
            accumulator -= FIXED_DT;
        }

        if (spherePBD) {
            spherePBD->applyToMesh();
        }


        glm::mat4 lightProjection = sun.getProjectionMatrix();
        glm::mat4 lightView = sun.getViewMatrix(camera.Position);
        glm::mat4 lightSpaceMatrix = lightProjection * lightView;

        shadowShader.use();
        shadowShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);

        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, depth.depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        for (map<Model*, vector<Entity*>>::iterator it = scene.entities.begin(); it != scene.entities.end(); ++it) {
            Model* model = it->first;
            for (Entity* entity : it->second) {
                shadowShader.setMat4("model", entity->getModelMatrix());
                model->bind();
                glDrawElements(GL_TRIANGLES, model->mesh.indices.size(), GL_UNSIGNED_INT, 0);
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

        lightingShader.use();

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        lightingShader.setMat4("projection", projection);
        lightingShader.setMat4("view", view);

        lightingShader.setVec3("viewPos", camera.Position);

        if (glm::length(sun.lightDir) < 0.1f) {
            sun.lightDir = glm::normalize(glm::vec3(0.5f, -1.0f, 0.3f));
        }
        lightingShader.setVec3("light.dir", sun.lightDir);
        lightingShader.setVec3("light.color", sun.lightColor);

        lightingShader.setFloat("useLighting", useLighting ? 1.0f : 0.0f);
        lightingShader.setFloat("useShadow", useShadow ? 1.0f : 0.0f);
        lightingShader.setFloat("filmTime", currentTime);

        lightingShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);
        lightingShader.setInt("shadowMap", 3);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, depth.ID);

        for (map<Model*, vector<Entity*>>::iterator it = scene.entities.begin(); it != scene.entities.end(); ++it) {
            Model* model = it->first;

            if (model == &yourOwnModel) continue;

            bool hasNormal = (model->normal != NULL);
            lightingShader.setFloat("useNormalMap", (useNormalMap && hasNormal) ? 1.0f : 0.0f);

            bool hasSpecular = (model->specular != NULL);
            lightingShader.setFloat("useSpecularMap", (useSpecular && hasSpecular) ? 1.0f : 0.0f);

            lightingShader.setFloat("usePCF", usePCF ? 1.0f : 0.0f);

            for (Entity* entity : it->second) {
                lightingShader.setMat4("world", entity->getModelMatrix());
                model->bind();
                glDrawElements(GL_TRIANGLES, model->mesh.indices.size(), GL_UNSIGNED_INT, 0);
            }
        }

        // use skybox Shader
        skyboxShader.use();
        glDepthFunc(GL_LEQUAL);
        view = glm::mat4(glm::mat3(camera.GetViewMatrix()));
        skyboxShader.setMat4("view", view);
        skyboxShader.setMat4("projection", projection);

        // render a skybox
        glBindVertexArray(VAOskybox);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture.textureID);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glBindVertexArray(0);
        glDepthFunc(GL_LESS);

        lightingShader.use();
        lightingShader.setFloat("filmTime", currentTime);
        lightingShader.setFloat("debugThickness", 1.0f);
        lightingShader.setFloat("useNormalMap", 0.0f);
        lightingShader.setFloat("useSpecularMap", 0.0f);
        lightingShader.setFloat("useShadow", 0.0f);
        lightingShader.setFloat("filmThicknessScale", 10000.0f);
        lightingShader.setFloat("filmRefractiveIndex", 1.34f);
        lightingShader.setFloat("filmAlpha", 0.1f);
        lightingShader.setFloat("filmR0", 0.025f);
        lightingShader.setFloat("filmDeltaMax", 1500.0f);
        lightingShader.setFloat("filmIridescenceStrength", 1.1f);
        lightingShader.setFloat("filmRefractionStrength", 0.8f);
        lightingShader.setFloat("filmFresnelStrength", 1.0f);
        lightingShader.setFloat("filmReflectionIntensity", 1.0f);
        lightingShader.setFloat("filmRoughness", 0.01f);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture.textureID);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, thinFilmLUT.ID);

        glDisable(GL_CULL_FACE);
        glDepthMask(GL_FALSE);

        auto itSpherePBD = scene.entities.find(&yourOwnModel);
        if (itSpherePBD != scene.entities.end()) {
            for (Entity* entity : itSpherePBD->second) {
                lightingShader.setMat4("world", entity->getModelMatrix());
                yourOwnModel.bind();
                glDrawElements(GL_TRIANGLES, yourOwnModel.mesh.indices.size(), GL_UNSIGNED_INT, 0);
            }
        }

        glDepthMask(GL_TRUE);
        lightingShader.setFloat("debugThickness", 0.0f);
        lightingShader.setFloat("useShadow", useShadow ? 1.0f : 0.0f);

        if (showWireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDisable(GL_CULL_FACE);
            glLineWidth(1.0f);

            // 스카이박스에서 유실된 카메라 뷰 행렬을 다시 온전하게 받아옵니다.
            view = camera.GetViewMatrix();

            wireframeShader.use();
            wireframeShader.setMat4("projection", projection);
            wireframeShader.setMat4("view", view); // 정상적인 뷰 행렬 적용

            auto itSphere = scene.entities.find(&yourOwnModel);
            if (itSphere != scene.entities.end()) {
                for (Entity* entity : itSphere->second) {
                    wireframeShader.setMat4("model", entity->getModelMatrix());
                    yourOwnModel.bind();
                    glDrawElements(GL_TRIANGLES, yourOwnModel.mesh.indices.size(), GL_UNSIGNED_INT, 0);
                }
            }

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}



// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window, DirectionalLight* sun)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS && !isKeyboardDone[GLFW_KEY_F]) {
        showWireframe = !showWireframe;
        isKeyboardDone[GLFW_KEY_F] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_RELEASE) {
        isKeyboardDone[GLFW_KEY_F] = false;
    }

    float t = 20.0f * deltaTime;
    

    bool lightChanged = false;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        sun->elevation += t;
        if (sun->elevation > 80.0f) sun->elevation = 80.0f;
        lightChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        sun->elevation -= t;
        if (sun->elevation < 15.0f) sun->elevation = 15.0f;
        lightChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        sun->azimuth += t;
        lightChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        sun->azimuth -= t;
        lightChanged = true;
    }

    if (lightChanged) {
        sun->updateLightDir();
    }

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}
