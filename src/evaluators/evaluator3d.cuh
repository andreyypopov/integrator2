#ifndef EVALUATOR3D_CUH
#define EVALUATOR3D_CUH

#include <vector>

#include "../common/gpu_timer.cuh"
#include "../NumericalIntegrator3d.cuh"
#include "../Mesh3d.cuh"

enum class output_format_enum {
    plainText = 1,
    csv = 2
};

class Evaluator3D
{
public:
    Evaluator3D(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_);
    virtual ~Evaluator3D();

    virtual void integrateOverSimpleNeighbors() = 0;
    virtual void integrateOverAttachedNeighbors() = 0;
    virtual void integrateOverNotNeighbors() = 0;

    virtual void runAllPairs(bool checkCorrectness = false);

    void runPairs(const std::vector<int3> &userSimpleNeighborsTasks, const std::vector<int3> &userAttachedNeighborsTasks, const std::vector<int3> &userNotNeighborsTasks);

    bool outputResultsToFile(neighbour_type_enum neighborType, output_format_enum outputFormat) const;

    const auto &getTasks(neighbour_type_enum neighborType) const {
        switch(neighborType){
            case neighbour_type_enum::simple_neighbors:
                return simpleNeighborsTasks;
            case neighbour_type_enum::attached_neighbors:
                return attachedNeighborsTasks;
            case neighbour_type_enum::not_neighbors:
                return notNeighborsTasks;
        }
    }

protected:
    int compareIntegrationResults(neighbour_type_enum neighborType, bool allPairs = false);

    deviceVector<int3> simpleNeighborsTasks;
    deviceVector<int3> attachedNeighborsTasks;
    deviceVector<int3> notNeighborsTasks;

    //separate values of theta and psi obtained after integration over Ki
    //of both regular and singular parts
    deviceVector<double4> d_simpleNeighborsIntegrals;
    deviceVector<double4> d_attachedNeighborsIntegrals;
    deviceVector<double4> d_notNeighborsIntegrals;

    //additional buffers for previous results of numerical integration (used for comparison of 2 refinement steps)
    deviceVector<double4> d_tempSimpleNeighborsIntegrals;
    deviceVector<double4> d_tempAttachedNeighborsIntegrals;
    deviceVector<double4> d_tempNotNeighborsIntegrals;

    //indices of original tasks which have not yet converged and are left for further integration
    //(lists for the next and the current iteration)
    deviceVector<int> simpleNeighborsTasksRest;
    deviceVector<int> attachedNeighborsTasksRest;
    deviceVector<int> notNeighborsTasksRest;
    deviceVector<int> tempSimpleNeighborsTasksRest;
    deviceVector<int> tempAttachedNeighborsTasksRest;
    deviceVector<int> tempNotNeighborsTasksRest;
    int *d_restTaskCount = nullptr;

    deviceVector<Point3> d_simpleNeighborsResults;
    deviceVector<Point3> d_attachedNeighborsResults;
    deviceVector<Point3> d_notNeighborsResults;

    const Mesh3D &mesh;
    NumericalIntegrator3D &numIntegrator;

    GpuTimer timer;

private:
    deviceVector<double> simpleNeighborsErrors;
    deviceVector<double> attachedNeighborsErrors;
    deviceVector<double> notNeighborsErrors;
};

#endif // EVALUATOR3D_CUH
