#ifndef NUMERICAL_INTEGRATOR_3D_CUH
#define NUMERICAL_INTEGRATOR_3D_CUH

#include "common/constants.h"
#include "Mesh3d.cuh"
#include "QuadratureFormula3d.cuh"

__device__ double4 integrate4D(const double4 *functionValues);

__device__ void calculateQuadraturePoints(Point3 *quadraturePoints, const Point3 *vertices, const int3 &triangle);

class NumericalIntegrator3D
{
public:
    NumericalIntegrator3D(const Mesh3D &mesh_, const QuadratureFormula3D &qf_);
	virtual ~NumericalIntegrator3D();

	void prepareTasksAndRefineWholeMesh(const deviceVector<int2> &simpleNeighborsTasks, const deviceVector<int2> &attachedNeighborsTasks,
			const deviceVector<int2> &notNeighborsTasks, int refineLevel = 0);

	void gatherResults(deviceVector<double4> &results, neighbour_type_enum neighborType) const;

	int getGaussPointsNumber() const {
		return GaussPointsNum;
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
    const int GaussPointsNum;
    const Mesh3D &mesh;
    const QuadratureFormula3D &qf;

    deviceVector<Point3> refinedVertices;

    deviceVector<int3> refinedCells;
    deviceVector<double> refinedCellMeasures;

	//tasks for refined cells (i,j and index of the original non-refined task)
    deviceVector<int3> refinedSimpleNeighborsTasks;
    deviceVector<int3> refinedAttachedNeighborsTasks;
    deviceVector<int3> refinedNotNeighborsTasks;

    //intermediate results of numerical integration (can be used for comparison of 2 iterations)
    deviceVector<double4> d_simpleNeighborsResults;
    deviceVector<double4> d_attachedNeighborsResults;
    deviceVector<double4> d_notNeighborsResults;

	//buffers for mesh refinement
	deviceVector<Point3> tempVertices;
    deviceVector<int3> tempCells;
    deviceVector<double> tempCellMeasures;
};

#endif // NUMERICAL_INTEGRATOR_3D_CUH
