cmake_minimum_required(VERSION 3.18)
project(casas_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization for Release
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
endif()

# Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Eigen
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(Eigen)
endif()

# Main module
add_library(casas_cpp MODULE bindings.cpp)

target_link_libraries(casas_cpp PRIVATE pybind11::module Eigen3::Eigen)

# Important: Force output directly to project root
set_target_properties(casas_cpp PROPERTIES 
    PREFIX ""
    SUFFIX ".pyd"
    OUTPUT_NAME "casas_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

message(STATUS "✅ Build configured - casas_cpp.pyd will be placed in project root")