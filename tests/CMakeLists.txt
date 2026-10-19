include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.18.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Tworzenie pliku wykonywalnego z testami (np. test_chunk.cpp)
add_executable(picsim_tests 
    SimulationSteps/ParticleInjectorStrategy/test_PointSourceStrategy.cpp
    SimulationSteps/ParticleInjectorStrategy/test_VelocityDistributionPolicy.cpp
    SimulationSteps/test_ChargeDepositionStep.cpp
    SimulationSteps/test_FieldSolverStep.cpp
    SimulationSteps/test_ParticleColliderStep.cpp
    SimulationSteps/test_ParticleInjectorStep.cpp
    SimulationSteps/test_ParticlePusherStep.cpp
    test_Chunk.cpp
    test_GridField.cpp 
    test_ParticleStorage.cpp
    test_SimulationEngine.cpp 
    test_SimulationUtils.cpp
    test_Species.cpp
)

# Wymuszenie flag SYCL dla celu testowego
target_compile_options(picsim_tests PRIVATE -fsycl)
target_link_options(picsim_tests PRIVATE -fsycl)

# Linkowanie z naszą biblioteką rdzenia oraz GTest
target_link_libraries(picsim_tests PRIVATE 
    picsim_core 
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(picsim_tests)