cmake_minimum_required(VERSION 3.20)

# Wymuszenie kompilatora icpx (przed project())
set(CMAKE_CXX_COMPILER icpx)

project(PicSimulation CXX)

# Standard C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Flagi SYCL i optymalizacji dla Intel OneAPI
add_compile_options(-fsycl -O3 -g3)
add_link_options(-fsycl)

set(CMAKE_BUILD_TYPE Release)

# 1. Deklaracja plików nagłówkowych (dla IDE)
set(HEADERS
    include/Chunk.hpp
    include/ParticleStorage.hpp
    include/Species.hpp
)

# 2. Deklaracja plików źródłowych
set(SOURCES
    src/Chunk.cpp
    src/ParticleStorage.cpp
    src/Species.cpp
    src/Sphere.cpp  # lub apps/Sphere.cpp
)

# 3. Definicja pliku wykonywalnego
add_executable(sphere ${SOURCES} ${HEADERS})

# 4. Dołączenie katalogu include do ścieżki wyszukiwania nagłówków
target_include_directories(sphere PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)