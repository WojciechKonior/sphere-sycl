include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Tworzenie pliku wykonywalnego z testami (np. test_chunk.cpp)
add_executable(picsim_tests test_chunk.cpp)

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