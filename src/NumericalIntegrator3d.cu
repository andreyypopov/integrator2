#include "NumericalIntegrator3d.cuh"

#include "common/cuda_memory.cuh"

NumericalIntegrator3D::NumericalIntegrator3D(const Mesh3D &mesh_, const QuadratureFormula3D &qf_)
    : GaussPointsNum(qf_.weights.size()), mesh(mesh_), qf(qf_)
{
    std::vector<double3> lCoordinates(GaussPointsNum);

    for(int i = 0; i < GaussPointsNum; ++i){
        lCoordinates[i].x = qf_.coordinates[i].x;
        lCoordinates[i].y = qf_.coordinates[i].y;
        lCoordinates[i].z = 1.0 - qf_.coordinates[i].x - qf_.coordinates[i].y;
    }

    copy_h2const(c_GaussPointsCoordinates, lCoordinates.data(), GaussPointsNum);
    copy_h2const(c_GaussPointsWeights, qf.weights.data(), GaussPointsNum);
}
