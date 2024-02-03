#ifndef EVALUATOR3D_CUH
#define EVALUATOR3D_CUH

#include <vector>

#include "../NumericalIntegrator3d.cuh"
#include "../Mesh3d.cuh"

enum class error_control_type_enum {
    fixed_refinement_level = 0,
    automatic_error_control = 1
};

class Evaluator3D
{
public:
    Evaluator3D(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_);
    virtual ~Evaluator3D();

    virtual void integrateOverSimpleNeighbors() = 0;
    virtual void integrateOverAttachedNeighbors() = 0;
    virtual void integrateOverNotNeighbors() = 0;

    virtual void runAllPairs();

    void setFixedRefinementLevel(int refinementLevel = 0);

protected:
    deviceVector<int2> simpleNeighborsTasks;
    deviceVector<int2> attachedNeighborsTasks;
    deviceVector<int2> notNeighborsTasks;

    //separate values of theta and psi obtained after integration over Ki
    //of both regular and singular parts
    deviceVector<double4> d_simpleNeighborsIntegrals;
    deviceVector<double4> d_attachedNeighborsIntegrals;
    deviceVector<double4> d_notNeighborsIntegrals;

    deviceVector<Point3> d_simpleNeighborsResults;
    deviceVector<Point3> d_attachedNeighborsResults;
    deviceVector<Point3> d_notNeighborsResults;

    std::vector<Point3> simpleNeighborsResults;
    std::vector<Point3> attachedNeighborsResults;
    std::vector<Point3> notNeighborsResults;

    const Mesh3D &mesh;
    NumericalIntegrator3D &numIntegrator;

    error_control_type_enum errorControlType;
    int meshRefinementLevel;
};

#endif // EVALUATOR3D_CUH
