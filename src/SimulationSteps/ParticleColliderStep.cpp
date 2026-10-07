#include "SimulationSteps/ParticleColliderStep.hpp"

sycl::event ParticleColliderStep::execute(SimulationContext &ctx, sycl::event dependency)
{
    sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh)
                                       {
            cgh.depends_on(dependency);
            cgh.single_task([=]() {
                // Kod MCC na GPU
            }); });

    std::cout << "  [" << name() << "] Przeliczono losowe zderzenia cząstek.\n";
    return evt;
}

std::string ParticleColliderStep::name() const { return "Particle Collider (MCC)"; }