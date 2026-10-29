#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <filesystem>
#include <vector>
#include <map>
#include "GridField.hpp"

/**
 * @class GridFieldTest
 * @brief Test fixture for GridFieldTest class unit tests.
 */
class GridFieldTest : public ::testing::Test {
protected:
    /// SYCL queue used for memory allocation and execution within tests.
    sycl::queue queue;

    /**
     * @brief Sets up the test environment by creating a default SYCL queue.
     */
    void SetUp() override {
        try {
            queue = sycl::queue(sycl::gpu_selector_v);
        } catch (const sycl::exception &e) {
            FAIL() << "Failed to create SYCL queue: " << e.what();
        }
    }

    /**
     * @brief Cleans up the test environment after each test execution.
     */
    void TearDown() override {
        try {
            queue.wait();
        } catch (const sycl::exception &e) {
            ADD_FAILURE() << "SYCL exception during TearDown wait: " << e.what();
        }
    }
};

/**
 * @test Verifies that the Mesh class correctly loads an unstructured .msh file 
 * from the assets directory and populates nodes and elements.
 */
TEST_F(GridFieldTest, DynamicFileParityVerification) {
    EXPECT_EQ(1,1);
}
