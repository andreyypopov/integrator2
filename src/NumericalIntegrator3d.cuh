#ifndef NUMERICAL_INTEGRATOR_3D_CUH
#define NUMERICAL_INTEGRATOR_3D_CUH

#include "common/constants.h"
#include "Mesh3d.cuh"
#include "QuadratureFormula3d.cuh"

enum class error_control_type_enum {
    fixed_refinement_level = 0,
    automatic_error_control = 1
};

__device__ double4 integrate4D(const double4 *functionValues);

__device__ void calculateQuadraturePoints(Point3 *quadraturePoints, const Point3 *vertices, const int3 &triangle);

class NumericalIntegrator3D
{
public:
    NumericalIntegrator3D(const Mesh3D &mesh_, const QuadratureFormula3D &qf_);
	virtual ~NumericalIntegrator3D();

	void setFixedRefinementLevel(int refinementLevel = 0);

	void prepareTasksAndMesh(const deviceVector<int3> &simpleNeighborsTasks, const deviceVector<int3> &attachedNeighborsTasks,	const deviceVector<int3> &notNeighborsTasks);

	void gatherResults(deviceVector<double4> &results, neighbour_type_enum neighborType) const;

	void refineMesh();

	int getGaussPointsNumber() const {
		return GaussPointsNum;
	}

	error_control_type_enum getErrorControlType() const {
		return errorControlType;
	}

    const auto &getRefinedSimpleNeighborsTasks() const {
        return refinedSimpleNeighborsTasks;
    }

    const auto &getSimpleNeighborsResults() const {
        return d_simpleNeighborsResults;
    }

    const auto &getRefinedAttachedNeighborsTasks() const {
        return refinedAttachedNeighborsTasks;
    }

    const auto &getAttachedNeighborsResults() const {
        return d_attachedNeighborsResults;
    }

    const auto &getRefinedNotNeighborsTasks() const {
        return refinedNotNeighborsTasks;
    }

    const auto &getNotNeighborsResults() const {
        return d_notNeighborsResults;
    }

    const auto &getRefinedVertices() const {
        return refinedVertices;
    }

    const auto &getRefinedCells() const {
        return refinedCells;
    }

	const auto &getRefinedCellMeasures() const {
        return refinedCellMeasures;
    }

private:
    int updateTasks(const deviceVector<int3> &originalTasks, neighbour_type_enum neighborType, bool performAllocation = false);
    
    const int GaussPointsNum;
    const Mesh3D &mesh;
    const QuadratureFormula3D &qf;

    deviceVector<Point3> refinedVertices;

    int2 verticesCellsNum;
    int2 *refinedVerticesCellsNum;
    deviceVector<int3> refinedCells;
    deviceVector<double> refinedCellMeasures;
    deviceVector<int> originalCells;    //vector of indices of original triangles (with respect to the refined triangles)

	//tasks for refined cells (i,j and index of the original non-refined task)
    deviceVector<int3> refinedSimpleNeighborsTasks;
    deviceVector<int3> refinedAttachedNeighborsTasks;
    deviceVector<int3> refinedNotNeighborsTasks;

    //intermediate results of numerical integration (for each pair of neighbors considering refined cells)
    deviceVector<double4> d_simpleNeighborsResults;
    deviceVector<double4> d_attachedNeighborsResults;
    deviceVector<double4> d_notNeighborsResults;

	//additional buffers for previous results of numerical integration (used for comparison of 2 refinement steps)
    deviceVector<double4> d_tempSimpleNeighborsResults;
    deviceVector<double4> d_tempAttachedNeighborsResults;
    deviceVector<double4> d_tempNotNeighborsResults;

    //additional buffers for tasks (in case of adaptive error control)
    deviceVector<int3> tempRefinedSimpleNeighborsTasks;
    deviceVector<int3> tempRefinedAttachedNeighborsTasks;
    deviceVector<int3> tempRefinedNotNeighborsTasks;

	//buffers for mesh refinement
	deviceVector<Point3> tempVertices;
    deviceVector<int3> tempCells;
    deviceVector<double> tempCellMeasures;
    deviceVector<int> tempOriginalCells;
    deviceVector<int> cellsToBeRefined;
    int *taskCount;

	error_control_type_enum errorControlType;
    int meshRefinementLevel;
};

#endif // NUMERICAL_INTEGRATOR_3D_CUH
