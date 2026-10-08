#include "Chunk.hpp"

Chunk::Chunk(sycl::queue &q, size_t capacity)
    : q_(q), capacity_(capacity), count_(0),
      d_posX_(nullptr), d_posY_(nullptr), d_posZ_(nullptr),
      d_velX_(nullptr), d_velY_(nullptr), d_velZ_(nullptr)
{
    if (capacity_ == 0)
    {
        return;
    }

    size_t bytes = capacity_ * sizeof(float);

    try
    {
        d_posX_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
        d_posY_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
        d_posZ_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
        d_velX_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
        d_velY_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
        d_velZ_ = static_cast<float *>(sycl::malloc_device(bytes, q_));
        d_dead_indices_ = sycl::malloc_device<int32_t>(capacity_, q_);
        d_dead_count_ = sycl::malloc_device<int32_t>(1, q_);
        q_.memset(d_dead_count_, 0, sizeof(int32_t)).wait();

        if (!d_posX_ || !d_posY_ || !d_posZ_ || !d_velX_ || !d_velY_ || !d_velZ_ || !d_dead_count_ || !d_dead_indices_)
        {
            throw std::bad_alloc();
        }
    }
    catch (...)
    {
        release(q_);
        throw;
    }
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

Chunk &Chunk::operator=(Chunk &&other) noexcept
{
    if (this != &other)
    {
        release(q_);

        q_ = other.q_;
        capacity_ = other.capacity_;
        count_ = other.count_;

        d_posX_ = std::exchange(other.d_posX_, nullptr);
        d_posY_ = std::exchange(other.d_posY_, nullptr);
        d_posZ_ = std::exchange(other.d_posZ_, nullptr);
        d_velX_ = std::exchange(other.d_velX_, nullptr);
        d_velY_ = std::exchange(other.d_velY_, nullptr);
        d_velZ_ = std::exchange(other.d_velZ_, nullptr);
    }
    return *this;
}

void Chunk::release(sycl::queue &q)
{
    try
    {
        q.wait();
        if (d_posX_)
        {
            sycl::free(d_posX_, q);
            d_posX_ = nullptr;
        }
        if (d_posY_)
        {
            sycl::free(d_posY_, q);
            d_posY_ = nullptr;
        }
        if (d_posZ_)
        {
            sycl::free(d_posZ_, q);
            d_posZ_ = nullptr;
        }
        if (d_velX_)
        {
            sycl::free(d_velX_, q);
            d_velX_ = nullptr;
        }
        if (d_velY_)
        {
            sycl::free(d_velY_, q);
            d_velY_ = nullptr;
        }
        if (d_velZ_)
        {
            sycl::free(d_velZ_, q);
            d_velZ_ = nullptr;
        }
        if (d_dead_indices_)
        {
            sycl::free(d_dead_indices_, q_);
            d_dead_indices_ = nullptr;
        }
        if (d_dead_count_)
        {
            sycl::free(d_dead_count_, q_);
            d_dead_count_ = nullptr;
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

void Chunk::set_count(size_t count) 
{ 
    count_ = count; 
}

size_t Chunk::available_space() const
{
    return capacity_ - count_;
}

ChunkSoAView Chunk::get_view() const
{
    return ChunkSoAView{
        d_posX_, d_posY_, d_posZ_,
        d_velX_, d_velY_, d_velZ_,
        count_, d_dead_indices_, d_dead_count_};
}