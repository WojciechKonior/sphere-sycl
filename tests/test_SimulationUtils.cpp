#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include "SimulationUtils.hpp"

/**
 * @class SimulationUtilsTest
 * @brief Test fixture for SimulationUtils helper functions and random number generation.
 */
class SimulationUtilsTest : public ::testing::Test {
protected:
    sycl::queue queue_;

    SimulationUtilsTest() : queue_(sycl::default_selector_v) {}

    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @brief Single comprehensive test verifying pcg_random_float properties:
 *        - Determinism (identical seeds and salts produce exact same values).
 *        - Range validation (all generated values fall strictly within [0.0, 1.0)).
 *        - Statistical convergence (mean of a large sample batch converges to 0.5).
 */
TEST_F(SimulationUtilsTest, PcgRandomFloatProperties) {
    // 1. Verify determinism
    uint64_t test_seed = 12345ULL;
    uint32_t test_salt = 7U;
    
    float first_val = SimulationUtils::pcg_random_float(test_seed, test_salt);
    float second_val = SimulationUtils::pcg_random_float(test_seed, test_salt);
    
    EXPECT_EQ(first_val, second_val);

    // 2. Verify range bounds and statistical mean over a batch of 10,000 samples
    const size_t num_samples = 10000;
    double sum = 0.0;
    size_t out_of_bounds_count = 0;

    for (size_t i = 0; i < num_samples; ++i) {
        uint64_t seed = static_cast<uint64_t>(i) + 1;
        uint32_t salt = static_cast<uint32_t>(i % 13);
        
        float r = SimulationUtils::pcg_random_float(seed, salt);

        // Check strict interval [0.0, 1.0)
        if (r < 0.0f || r >= 1.0f) {
            out_of_bounds_count++;
        }
        sum += static_cast<double>(r);
    }

    // No value can violate the bounds
    EXPECT_EQ(out_of_bounds_count, 0ul);

    // For a uniform distribution on [0, 1), the expected mean is 0.5
    double empirical_mean = sum / static_cast<double>(num_samples);
    EXPECT_NEAR(empirical_mean, 0.5, 0.01);
}