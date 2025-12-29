#include "distributed/job_pool.hpp"
#include <gtest/gtest.h>

using namespace tnn;

TEST(JobPoolTest, TensorAllocation) {
  Tensor<float> t({1, 1, 10, 10});
  EXPECT_GE(t.capacity(), 100);
}

TEST(JobPoolTest, BasicFunctionality) {
  auto &pool = JobPool<float>::instance();

  {
    auto job = pool.get_job(100);
    EXPECT_TRUE(job != nullptr);
    EXPECT_EQ(job->data.capacity(), 0);
    job->data = Tensor<float>({1, 1, 10, 10});
    EXPECT_GE(job->data.capacity(), 100);
  }

  EXPECT_EQ(pool.pool_size(), 1);

  {
    auto job = pool.get_job(50);
    EXPECT_GE(job->data.capacity(), 100);
    EXPECT_EQ(pool.pool_size(), 0);
  }

  EXPECT_EQ(pool.pool_size(), 1);
}

TEST(JobPoolTest, MultipleJobs) {
  auto &pool = JobPool<float>::instance();

  std::vector<PooledJob<float>> jobs;
  for (int i = 0; i < 10; ++i) {
    jobs.push_back(pool.get_job(100));
    jobs.back()->data = Tensor<float>({1, 1, 10, 10});
  }
}
