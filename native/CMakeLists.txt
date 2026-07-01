cmake_minimum_required(VERSION 3.26)

project(yurameki_cuda_collide LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_library(yurameki_cuda_collide SHARED
    yurameki_cuda_collide.cu
    yurameki_cuda_collide.h
)

target_compile_definitions(yurameki_cuda_collide PRIVATE YURAMEKI_CUDA_BUILD=1)
set_target_properties(yurameki_cuda_collide PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    OUTPUT_NAME "yurameki_cuda_collide"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
