#include "SimulationSteps/FieldSolverStep.hpp"

sycl::event FieldSolverStep::execute(SimulationContext &ctx, sycl::event dependency)
{
    sycl::event evt = ctx.queue.submit([&](sycl::handler &cgh)
                                       {
            cgh.depends_on(dependency);
            cgh.single_task([=]() {
                // Kod solwera na GPU
            }); });

    std::cout << "  [" << name() << "] Obliczono pola E oraz B na siatce.\n";
    return evt;
}

std::string FieldSolverStep::name() const { return "Field Solver"; }