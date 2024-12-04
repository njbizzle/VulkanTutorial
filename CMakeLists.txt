cmake_minimum_required(VERSION 3.10.0)
project(VulkanTutorial VERSION 0.1.0 LANGUAGES CXX)

set(GLFW_DIR ${CMAKE_SOURCE_DIR}/dependencies/glfw)
set(VULKAN_DIR ${CMAKE_SOURCE_DIR}/dependencies/vulkan)
set(STB_DIR ${CMAKE_SOURCE_DIR}/dependencies/stb)

add_executable(${PROJECT_NAME} src/main.cpp)

include_directories(${GLFW_DIR}/include)
include_directories(${VULKAN_DIR}/include)
include_directories(${STB_DIR}/include)

set(GLFW_LIB "${GLFW_DIR}/lib/libglfw.3.dylib")
set(VULKAN_LIB "${VULKAN_DIR}/lib/libvulkan.dylib")

target_link_libraries(${PROJECT_NAME} ${GLFW_LIB} ${VULKAN_LIB})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
