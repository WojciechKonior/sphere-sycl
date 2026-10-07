#include "Chunk.hpp"

Chunk::Chunk(sycl::queue &q, size_t capacity)
    : q_(q), capacity_(capacity), count_(0)
{
    size_t bytes = capacity_ * sizeof(float);
    d_posX_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
    d_posY_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
    d_posZ_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
    d_velX_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
    d_velY_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
    d_velZ_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
}

Chunk::~Chunk()
{
    release(q_);
};

Chunk::Chunk(Chunk &&other) noexcept
    : q_(other.q_), capacity_(other.capacity_), count_(other.count_),
      d_posX_(std::exchange(other.d_posX_, nullptr)),
      d_posY_(std::exchange(other.d_posY_, nullptr)),
      d_posZ_(std::exchange(other.d_posZ_, nullptr)),
      d_velX_(std::exchange(other.d_velX_, nullptr)),
      d_velY_(std::exchange(other.d_velY_, nullptr)),
      d_velZ_(std::exchange(other.d_velZ_, nullptr)) {}

void Chunk::release(sycl::queue &q)
{
    try
    {
        if (d_posX_)
        {
            sycl::free(d_posX_, q_);
            d_posX_ = nullptr;
        }
        if (d_posY_)
        {
            sycl::free(d_posY_, q_);
            d_posY_ = nullptr;
        }
        if (d_posZ_)
        {
            sycl::free(d_posZ_, q_);
            d_posZ_ = nullptr;
        }
        if (d_velX_)
        {
            sycl::free(d_velX_, q_);
            d_velX_ = nullptr;
        }
        if (d_velY_)
        {
            sycl::free(d_velY_, q_);
            d_velY_ = nullptr;
        }
        if (d_velZ_)
        {
            sycl::free(d_velZ_, q_);
            d_velZ_ = nullptr;
        }
    }
    catch (const sycl::exception &e)
    {
        std::cerr << "SYCL free error: " << e.what() << std::endl;
    }
}

size_t Chunk::capacity() const
{
    return capacity_;
}

size_t Chunk::count() const
{
    return count_;
}

size_t Chunk::available_space() const
{
    return capacity_ - count_;
}