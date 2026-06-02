#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuseSampler;
    sampler2D specularSampler;
    sampler2D normalSampler;
    float shininess;
}; 

struct Light {
    vec3 dir;
    vec3 color; // this is I_d (I_s = I_d, I_a = 0.3 * I_d)
};

uniform vec3 viewPos;
uniform Material material;
uniform Light light;

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in mat3 TBN;
in vec4 FragPosLightSpace;
in vec3 VertexColor;
in float FilmThickness;

uniform sampler2D shadowMap;
uniform samplerCube skyboxTexture;
uniform sampler2D thinFilmLUT;

uniform float useNormalMap;
uniform float useSpecularMap;
uniform float useShadow;
uniform float useLighting;

uniform float usePCF;

uniform float debugThickness;
uniform float filmTime;
uniform float filmThicknessScale;
uniform float filmRefractiveIndex;
uniform float filmAlpha;
uniform float filmR0;
uniform float filmDeltaMax;
uniform float filmIridescenceStrength;
uniform float filmRefractionStrength;
uniform float filmFresnelStrength;
uniform float filmReflectionIntensity;
uniform float filmRoughness;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

const vec2 poissonDisk[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870), vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790)
);

float schlickFresnel(float cosTheta, float r0)
{
    float oneMinusCos = 1.0 - clamp(cosTheta, 0.0, 1.0);
    return r0 + (1.0 - r0) * pow(oneMinusCos, 5.0);
}

float distributionGGX(float NoH, float roughness)
{
    float a = max(roughness * roughness, 0.001);
    float a2 = a * a;
    float denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denom * denom, 1e-5);
}

float visibilitySmithGGXCorrelated(float NoV, float NoL, float roughness)
{
    float a = max(roughness * roughness, 0.001);
    float gv = NoL * sqrt(NoV * (NoV - NoV * a) + a);
    float gl = NoV * sqrt(NoL * (NoL - NoL * a) + a);
    return 0.5 / max(gv + gl, 1e-5);
}

float opticalPathDifference(float h, float cosTheta, float n)
{
    float sin2Theta = max(0.0, 1.0 - cosTheta * cosTheta);
    float refracted = sqrt(max(0.0, 1.0 - sin2Theta / (n * n)));
    return 2.0 * n * h * refracted;
}

vec3 sampleThinFilmLUT(float hNm, float cosTheta)
{
    // U 좌표에 deltaNm 대신 두께(hNm)를 직접 사용합니다.
    vec2 uv = vec2(clamp(hNm / filmDeltaMax, 0.0, 1.0), clamp(cosTheta, 0.0, 1.0));
    vec3 lutColor = texture(thinFilmLUT, uv).rgb;
    return lutColor;
}

float continuousFilmThickness(float h, vec3 normal, vec3 worldPos)
{
    return max(0.001, h);
}

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    if(projCoords.z > 1.0) return 0.0;

    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.0005);

    if (usePCF < 0.5f) {
        float closestDepth = texture(shadowMap, projCoords.xy).r; 
        float shadow = projCoords.z - bias > closestDepth ? 1.0 : 0.0;
        return shadow;
    }

    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);

    float r = rand(gl_FragCoord.xy) * 6.28318530718;
    float s = sin(r);
    float c = cos(r);
    mat2 rot = mat2(c, -s, s, c); 

    float shadowRadius = 2.0; 

    for(int i = 0; i < 16; ++i) {
        vec2 offset = rot * poissonDisk[i] * texelSize * shadowRadius;
        float pcfDepth = texture(shadowMap, projCoords.xy + offset).r; 
        shadow += projCoords.z - bias > pcfDepth ? 1.0 : 0.0;        
    }
    
    shadow /= 16.0;
    return shadow;
}



void main()
{
    vec3 color = texture(material.diffuseSampler, TexCoord).rgb;

    // on-off by key 3 (useLighting). 
    // if useLighting is 0, return diffuse value without considering any lighting.(DO NOT CHANGE)
    if (useLighting < 0.5f){
        FragColor = vec4(color, 1.0); 
        return; 
    }

    vec3 normal = normalize(Normal);
    vec3 lightDir = normalize(-light.dir);

    if(useNormalMap > 0.5f)
    {
        normal = texture(material.normalSampler, TexCoord).rgb;
        normal = normalize(normal * 2.0 - 1.0);
        normal = normalize(TBN * normal);
    }

    float shadow = 0.0;
    if (useShadow > 0.5f) {
        shadow = ShadowCalculation(FragPosLightSpace, normal, lightDir);
    }

    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * light.color * color; 
    vec3 ambient = 0.3 * light.color * color;

    vec3 specular = vec3(0.0);

    if(useSpecularMap > 0.5f)
    {
        float k_s = texture(material.specularSampler, TexCoord).r;
        
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 halfwayDir = normalize(lightDir + viewDir);  
        float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
        
        specular = light.color * spec * k_s;
    }
    
    vec3 viewDir = normalize(viewPos - FragPos);
    float cosTheta = clamp(dot(normal, viewDir), 0.0, 1.0);
    float fresnel = schlickFresnel(cosTheta, filmR0);
    vec3 result;
    
    if (debugThickness > 0.5f) {
        float opticalThickness = continuousFilmThickness(FilmThickness, normal, FragPos);
        float hNm = opticalThickness * filmThicknessScale;
        float deltaNm = opticalPathDifference(hNm, cosTheta, filmRefractiveIndex);
        vec3 interferenceRGB = sampleThinFilmLUT(hNm, cosTheta);
        vec3 reflectedDir = reflect(-viewDir, normal);
        vec3 envReflection = texture(skyboxTexture, reflectedDir).rgb;
        vec3 refractedDir = refract(-viewDir, normal, 1.0 / filmRefractiveIndex);
        if (length(refractedDir) < 1e-4) {
            refractedDir = reflectedDir;
        }
        vec3 envRefraction = texture(skyboxTexture, refractedDir).rgb;

        vec3 halfDir = normalize(lightDir + viewDir);
        float NoL = clamp(dot(normal, lightDir), 0.0, 1.0);
        float NoV = max(cosTheta, 1e-4);
        float NoH = clamp(dot(normal, halfDir), 0.0, 1.0);
        float LoH = clamp(dot(lightDir, halfDir), 0.0, 1.0);

        float directionalFresnel = schlickFresnel(LoH, filmR0) * filmFresnelStrength;
        float D = distributionGGX(NoH, filmRoughness);
        float V = visibilitySmithGGXCorrelated(NoV, NoL, filmRoughness);
        float cookTorranceSpecular = D * V * NoL;

        float viewFresnel = fresnel * filmFresnelStrength;
        float localIridescence = clamp(cookTorranceSpecular * directionalFresnel * filmIridescenceStrength, 0.0, 3.0);
        float envIridescence = viewFresnel * filmReflectionIntensity;

        vec3 transparentFilm = envRefraction * filmRefractionStrength * (1.0 - clamp(viewFresnel, 0.0, 0.85));
        vec3 neutralReflection = envReflection * viewFresnel * 0.18;
        vec3 iridescentEnv = envReflection * interferenceRGB * envIridescence;
        vec3 iridescentLight = light.color * interferenceRGB * localIridescence;
        result = transparentFilm + neutralReflection + iridescentEnv + iridescentLight;
    } else {
        result = ambient + diffuse + specular;
    }

    float alpha = debugThickness > 0.5f ? clamp(filmAlpha + fresnel * 0.36 * filmFresnelStrength, filmAlpha, 0.70) : 1.0;
    FragColor = vec4(result, alpha);

}
