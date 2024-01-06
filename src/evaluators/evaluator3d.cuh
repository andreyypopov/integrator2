#ifndef EVALUATOR3D_CUH
#define EVALUATOR3D_CUH

#include <vector>

#include "../NumericalIntegrator3d.cuh"
#include "../Mesh3d.cuh"

class Evaluator3D
{
public:
    Evaluator3D(const Mesh3D &mesh_, const NumericalIntegrator3D* const numIntegrator_ = nullptr);
    virtual ~Evaluator3D();

    virtual void integrateOverSimpleNeighbors() = 0;
    virtual void integrateOverAttachedNeighbors() = 0;
    virtual void integrateOverNotNeighbors() = 0;

    virtual void runAllPairs();

protected:
    int simpleNeighborsTasksNum = 0;
    int attachedNeighborsTasksNum = 0;
    int notNeighborsTasksNum = 0;
    int2 *simpleNeighborsTasks;
    int2 *attachedNeighborsTasks;
    int2 *notNeighborsTasks;

    double *d_simpleNeighborsResults;
    double *d_attachedNeighborsResults;
    double *d_notNeighborsResults;

    std::vector<double> simpleNeighborsResults;
    std::vector<double> attachedNeighborsResults;
    std::vector<double> notNeighborsResults;

    const Mesh3D &mesh;
    const NumericalIntegrator3D* const numIntegrator;
};

#endif // EVALUATOR3D_CUH
