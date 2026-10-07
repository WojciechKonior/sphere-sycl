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
    include/SimulationSteps/ChargeDepositionStep.hpp
    include/SimulationSteps/FieldSolverStep.hpp
    include/SimulationSteps/ISimulationStep.hpp
    include/SimulationSteps/ParticleColliderStep.hpp
    include/SimulationSteps/ParticleInjectorStep.hpp
    include/SimulationSteps/ParticlePusherStep.hpp
    include/SimulationSteps/SimulationContext.hpp
    include/Chunk.hpp
    include/GridField.hpp
    include/ParticleStorage.hpp
    include/SimulationEngine.hpp
    include/Species.hpp
)

# 2. Deklaracja plików źródłowych rdzenia symulacji (BEZ main w Sphere.cpp)
set(CORE_SOURCES
    src/SimulationSteps/ChargeDepositionStep.cpp
    src/SimulationSteps/FieldSolverStep.cpp
    src/SimulationSteps/ParticleColliderStep.cpp
    src/SimulationSteps/ParticleInjectorStep.cpp
    src/SimulationSteps/ParticlePusherStep.cpp
    src/Chunk.cpp
    src/GridField.cpp
    src/ParticleStorage.cpp
    src/SimulationEngine.cpp
    src/Species.cpp
)

# 3. Tworzenie biblioteki rdzenia symulacji
add_library(picsim_core STATIC ${CORE_SOURCES} ${HEADERS})

# PUBLIC sprawia, że cele konsumujące (sphere oraz testy) automatycznie dostaną ścieżkę include
target_include_directories(picsim_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 4. Definicja głównej aplikacji symulacji
add_executable(sphere src/Sphere.cpp)
target_link_libraries(sphere PRIVATE picsim_core)

# 5. Włączenie i dołączenie podkatalogu z testami
enable_testing()
add_subdirectory(tests)