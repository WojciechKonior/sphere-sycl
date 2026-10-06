cmake_minimum_required(VERSION 3.20)

# Wymuszenie kompilatora icpx (MUSI być przed project())
set(CMAKE_CXX_COMPILER icpx)

project(PicSimulation CXX)

# Standard C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Flag SYCL i optymalizacji dla Intel OneAPI
add_compile_options(-fsycl -O3 -g3)
add_link_options(-fsycl)

set(CMAKE_BUILD_TYPE Release)

# Pliki źródłowe oraz nagłówkowe (dla IDE)
set(SOURCES
    ParticleStorage.cpp
    ParticleStorage.hpp
    Chunk.cpp
    Chunk.hpp
    Sphere.cpp
    Species.cpp
    Species.hpp
)

add_executable(sphere ${SOURCES})

target_include_directories(sphere PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})