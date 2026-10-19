cmake_minimum_required(VERSION 3.25)

# Wymuszenie kompilatora icpx (przed project())
set(CMAKE_CXX_COMPILER icpx)

project(PicSimulation CXX)

# Standard C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Konfiguracja kompilacji
set(CMAKE_BUILD_TYPE Release)

# Wybór backendu: NVIDIA, AMD lub INTEL
set(PICSIM_GPU "NVIDIA" CACHE STRING "GPU backend")
set_property(CACHE PICSIM_GPU PROPERTY STRINGS NVIDIA AMD INTEL)

# Architektura GPU (używana dla NVIDIA i AMD)
set(PICSIM_GPU_ARCH "sm_75" CACHE STRING "GPU architecture")

# Wspólne flagi SYCL
add_compile_options(-fsycl -O3 -g3 -Werror)
add_link_options(-fsycl)

# Konfiguracja zależna od producenta GPU
if(PICSIM_GPU STREQUAL "NVIDIA")

    message(STATUS "PicSimulation: NVIDIA CUDA backend")
    message(STATUS "GPU architecture: ${PICSIM_GPU_ARCH}")

    add_compile_options(
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend=nvptx64-nvidia-cuda
        --cuda-gpu-arch=${PICSIM_GPU_ARCH}
    )

    add_link_options(
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend=nvptx64-nvidia-cuda
        --cuda-gpu-arch=${PICSIM_GPU_ARCH}
    )

elseif(PICSIM_GPU STREQUAL "AMD")

    message(STATUS "PicSimulation: AMD HIP backend")
    message(STATUS "GPU architecture: ${PICSIM_GPU_ARCH}")

    add_compile_options(
        -fsycl-targets=amdgcn-amd-amdhsa
        -Xsycl-target-backend=amdgcn-amd-amdhsa
        --offload-arch=${PICSIM_GPU_ARCH}
    )

    add_link_options(
        -fsycl-targets=amdgcn-amd-amdhsa
        -Xsycl-target-backend=amdgcn-amd-amdhsa
        --offload-arch=${PICSIM_GPU_ARCH}
    )

elseif(PICSIM_GPU STREQUAL "INTEL")

    message(STATUS "PicSimulation: Intel GPU backend")

    add_compile_options(-fsycl-targets=spir64)
    add_link_options(-fsycl-targets=spir64)

else()

    message(FATAL_ERROR
        "Nieznany PICSIM_GPU='${PICSIM_GPU}'. "
        "Wybierz NVIDIA, AMD lub INTEL."
    )

endif()

# 1. Deklaracja plików nagłówkowych (dla IDE)
set(HEADERS
    include/SimulationSteps/ParticleInjectorStrategy/BeamSourceStrategy.hpp
    include/SimulationSteps/ParticleInjectorStrategy/IParticleInjectorStrategy.hpp
    include/SimulationSteps/ParticleInjectorStrategy/PointSourceStrategy.hpp
    include/SimulationSteps/ParticleInjectorStrategy/VelocityDistributionPolicy.hpp
    include/SimulationSteps/ChargeDepositionStep.hpp
    include/SimulationSteps/FieldSolverStep.hpp
    include/SimulationSteps/ISimulationStep.hpp
    include/SimulationSteps/ParticleColliderStep.hpp
    include/SimulationSteps/ParticleInjectorStep.hpp
    include/SimulationSteps/ParticlePusherStep.hpp
    include/Chunk.hpp
    include/GridField.hpp
    include/ParticleStorage.hpp
    include/SimulationEngine.hpp
    include/SimulationUtils.hpp
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