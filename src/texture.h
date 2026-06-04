#ifndef TEXTURE_H
#define TEXTURE_H
#define STB_IMAGE_IMPLEMENTATION   // use of stb functions once and for all
#include "stb_image.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
class Texture{
public:
    unsigned int ID;
    int width;
    int height;
    int channels;
    Texture() {}
    Texture(const char* filePath)
    {   
        glGenTextures(1, &ID);
        glBindTexture(GL_TEXTURE_2D, ID);
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_set_flip_vertically_on_load(true);
        
        unsigned char *data = stbi_load(filePath, &width, &height, &channels, 0);
        if (data)
        {   
            if(channels == 3){
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            }else{
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            }
            
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else
        {
            std::cout << "Failed to load texture" << std::endl;
        }
        stbi_image_free(data);
    }
};

class ThinFilmLUTTexture
{
public:
    unsigned int ID;
    int width;
    int height;

    ThinFilmLUTTexture(int lutWidth = 256, int lutHeight = 128)
        : ID(0), width(lutWidth), height(lutHeight)
    {
        std::vector<float> pixels(width * height * 4, 0.0f);

        const float filmIOR = 1.34f;
        const float incidentIOR = 1.0f;
        const float exitIOR = 1.0f;
        const int wavelengthSamples = 81;
        const float lambdaMin = 380.0f;
        const float lambdaMax = 780.0f;
        const float lambdaStep = (lambdaMax - lambdaMin) / float(wavelengthSamples - 1);

        auto clamp01 = [](float v) {
            return std::max(0.0f, std::min(1.0f, v));
        };

        auto wavelengthToXYZ = [](float lambdaNm) {
            const float x =
                1.056f * std::exp(-0.5f * std::pow((lambdaNm - 599.8f) / 37.9f, 2.0f)) +
                0.362f * std::exp(-0.5f * std::pow((lambdaNm - 442.0f) / 16.0f, 2.0f)) -
                0.065f * std::exp(-0.5f * std::pow((lambdaNm - 501.1f) / 20.4f, 2.0f));

            const float y =
                0.821f * std::exp(-0.5f * std::pow((lambdaNm - 568.8f) / 46.9f, 2.0f)) +
                0.286f * std::exp(-0.5f * std::pow((lambdaNm - 530.9f) / 16.3f, 2.0f));

            const float z =
                1.217f * std::exp(-0.5f * std::pow((lambdaNm - 437.0f) / 11.8f, 2.0f)) +
                0.681f * std::exp(-0.5f * std::pow((lambdaNm - 459.0f) / 26.0f, 2.0f));

            return glm::vec3(x, y, z);
        };

        auto xyzToLinearSRGB = [](const glm::vec3& xyz) {
            return glm::vec3(
                3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z,
                -0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z,
                0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z
            );
        };

        auto fresnelAmplitude = [](float n1, float n2, float cosTheta1, float cosTheta2, bool sPolarized) {
            const float eps = 1e-6f;
            const float a = std::max(n1 * cosTheta1, eps);
            const float b = std::max(n2 * cosTheta2, eps);
            if (sPolarized) {
                return (a - b) / (a + b);
            }
            const float c = std::max(n2 * cosTheta1, eps);
            const float d = std::max(n1 * cosTheta2, eps);
            return (c - d) / (c + d);
        };

        auto thinFilmReflectance = [&](float lambdaNm, float thicknessNm, float cosThetaIncident, bool sPolarized) {
            const float sinThetaIncident = std::sqrt(std::max(0.0f, 1.0f - cosThetaIncident * cosThetaIncident));
            const float sinThetaFilm = incidentIOR / filmIOR * sinThetaIncident;
            const float cosThetaFilm = std::sqrt(std::max(0.0f, 1.0f - sinThetaFilm * sinThetaFilm));

            const float sinThetaExit = filmIOR / exitIOR * sinThetaFilm;
            const float cosThetaExit = std::sqrt(std::max(0.0f, 1.0f - sinThetaExit * sinThetaExit));

            const float r01 = fresnelAmplitude(incidentIOR, filmIOR, cosThetaIncident, cosThetaFilm, sPolarized);
            const float r12 = fresnelAmplitude(filmIOR, exitIOR, cosThetaFilm, cosThetaExit, sPolarized);

            const float beta = 2.0f * 3.14159265358979323846f * filmIOR * thicknessNm * cosThetaFilm / lambdaNm;
            const std::complex<float> phase = std::polar(1.0f, 2.0f * beta);
            const std::complex<float> numerator(r01, 0.0f);
            const std::complex<float> numerator2(r12, 0.0f);
            const std::complex<float> denominator = std::complex<float>(1.0f, 0.0f) + numerator * numerator2 * phase;
            const std::complex<float> amplitude = (numerator + numerator2 * phase) / denominator;
            return std::norm(amplitude);
        };

        for (int y = 0; y < height; ++y) {
            float cosTheta = static_cast<float>(y) / static_cast<float>(height - 1);
            for (int x = 0; x < width; ++x) {
                float deltaNm = 1200.0f * static_cast<float>(x) / static_cast<float>(width - 1);
                const float thicknessNm = deltaNm;

                glm::vec3 rgb(0.0f);
                float transmittanceLuma = 0.0f;
                float spectralWeightSum = 0.0f;

                for (int i = 0; i < wavelengthSamples; ++i) {
                    const float lambdaNm = lambdaMin + lambdaStep * float(i);
                    const glm::vec3 xyzWeight = wavelengthToXYZ(lambdaNm);
                    const float weight = std::max(0.0f, xyzWeight.y);
                    if (weight <= 0.0f) {
                        continue;
                    }

                    const float rs = thinFilmReflectance(lambdaNm, thicknessNm, clamp01(cosTheta), true);
                    const float rp = thinFilmReflectance(lambdaNm, thicknessNm, clamp01(cosTheta), false);
                    const float reflectance = 0.5f * (rs + rp);
                    const float transmittance = std::max(0.0f, 1.0f - reflectance);

                    const glm::vec3 spectralRGB = xyzToLinearSRGB(xyzWeight);
                    rgb += spectralRGB * reflectance * weight;
                    transmittanceLuma += transmittance * weight;
                    spectralWeightSum += weight;
                }

                if (spectralWeightSum > 1e-6f) {
                    rgb /= spectralWeightSum;
                    transmittanceLuma /= spectralWeightSum;
                }

                rgb = glm::max(rgb, glm::vec3(0.0f));

                const int idx = (y * width + x) * 4;
                pixels[idx + 0] = clamp01(rgb.r);
                pixels[idx + 1] = clamp01(rgb.g);
                pixels[idx + 2] = clamp01(rgb.b);
                pixels[idx + 3] = clamp01(transmittanceLuma);
            }
        }

        glGenTextures(1, &ID);
        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, &pixels[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};

class DepthMapTexture 
{
public:
	unsigned int ID;
	unsigned int depthMapFBO;
	int width;
	int height;

	DepthMapTexture(int shadow_width, int shadow_height) 
	{
		width = shadow_width;
		height = shadow_height;
		glGenFramebuffers(1, &depthMapFBO);
		glGenTextures(1, &ID);
		glBindTexture(GL_TEXTURE_2D, ID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
		// attach depth texture as FBO's depth buffer
		glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, ID, 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
};
#endif
