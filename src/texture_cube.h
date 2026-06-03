#ifndef TEXTURE_CUBE_H
#define TEXTURE_CUBE_H
#include "texture.h"

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class CubemapTexture {
public:
    unsigned int textureID;
    int width;
    int height;
    int channels;

    CubemapTexture(std::vector<std::string> faces)
    {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

        // 큐브맵은 상하 반전을 하지 않는 것이 표준입니다.
        stbi_set_flip_vertically_on_load(false);

        for (unsigned int i = 0; i < faces.size(); i++)
        {
            unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &channels, 0);
            if (data)
            {
                // [수정 1] 이미지의 실제 채널 수에 맞춰 동적으로 포맷 설정 (.png 오류 방지)
                GLenum format = GL_RGB;
                if (channels == 1) format = GL_RED;
                else if (channels == 3) format = GL_RGB;
                else if (channels == 4) format = GL_RGBA;

                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
                stbi_image_free(data);
            }
            else
            {
                std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
                stbi_image_free(data);
            }
        }

        // [수정 2] 밉맵(Mipmap) 생성: 거리에 따른 화질 저하와 픽셀 깨짐을 완벽히 방지합니다.
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

        // [수정 3] 축소 필터를 밉맵 기반 선형 보간(LINEAR_MIPMAP_LINEAR)으로 변경하여 압도적으로 부드러운 화질 제공
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }
};
#endif