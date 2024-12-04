#!/bin/sh

dependencies/vulkan/bin/glslc shaders/shader.vert -o shaders/spirv/vert.spv
dependencies/vulkan/bin/glslc shaders/shader.frag -o shaders/spirv/frag.spv
