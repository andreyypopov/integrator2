#ifndef EVALUATOR3D_CUH
#define EVALUATOR3D_CUH

#include <vector>

#include "../NumericalIntegrator3d.cuh"
#include "../Mesh3d.cuh"

class Evaluator3D
{
public:
    Evaluator3D(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_);
    virtual ~Evaluator3D();

    virtual void integrateOverSimpleNeighbors() = 0;
    virtual void integrateOverAttachedNeighbors() = 0;
    virtual void integrateOverNotNeighbors() = 0;

    virtual void runAllPairs();

    bool outputResultsToFile(neighbour_type_enum neighborType) const;

protected:
    deviceVector<int3> simpleNeighborsTasks;
    deviceVector<int3> attachedNeighborsTasks;
    deviceVector<int3> notNeighborsTasks;

    //separate values of theta and psi obtained after integration over Ki
    //of both regular and singular parts
    deviceVector<double4> d_simpleNeighborsIntegrals;
    deviceVector<double4> d_attachedNeighborsIntegrals;
    deviceVector<double4> d_notNeighborsIntegrals;

    deviceVector<Point3> d_simpleNeighborsResults;
    deviceVector<Point3> d_attachedNeighborsResults;
    deviceVector<Point3> d_notNeighborsResults;

    const Mesh3D &mesh;
    NumericalIntegrator3D &numIntegrator;
};

#endif // EVALUATOR3D_CUH
