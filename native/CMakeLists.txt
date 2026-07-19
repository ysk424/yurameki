cmake_minimum_required(VERSION 3.24)

project(yurameki_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_yurameki_native_0_3_0 MODULE
    src/yurameki_native.cpp
)

target_link_libraries(_yurameki_native_0_3_0 PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(_yurameki_native_0_3_0 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

if(MSVC)
    target_compile_options(_yurameki_native_0_3_0 PRIVATE /O2 /fp:precise /W4 /EHsc)
else()
    target_compile_options(_yurameki_native_0_3_0 PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# Keep the normal development module beside the add-on package.  A separate
# output directory can be selected when a running Blender process has locked
# that file and an audit build is needed.
set(YURAMEKI_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
    "Directory for the built Python extension module")
set_target_properties(_yurameki_native_0_3_0 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${YURAMEKI_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${YURAMEKI_OUTPUT_DIRECTORY}"
)
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(_yurameki_native_0_3_0 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${YURAMEKI_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${YURAMEKI_OUTPUT_DIRECTORY}"
    )
endforeach()
