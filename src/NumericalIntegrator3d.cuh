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

	void refineMesh(neighbour_type_enum updateTasksNeighborType = neighbour_type_enum::undefined);

    void resetMesh();

    int determineCellsToBeRefined(deviceVector<int> &restTasks, const deviceVector<int3> &tasks, neighbour_type_enum neighborType);

	int getGaussPointsNumber() const {
		return GaussPointsNum;
	}

    int getQuadratureFormulaOrder() const {
        return qf.order;
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

    auto &getCellsToBeRefined() const {
        return cellsToBeRefined;
    }

    auto &getIntegralsConverged(neighbour_type_enum neighborType){
        switch(neighborType){
            case neighbour_type_enum::simple_neighbors:
                return simpleNeighborsIntegralsConverged;
            case neighbour_type_enum::attached_neighbors:
                return attachedNeighborsIntegralsConverged;
            case neighbour_type_enum::not_neighbors:
                return notNeighborsIntegralsConverged;
        }
    }

    auto &getRefinementsRequired(neighbour_type_enum neighborType){
        switch(neighborType){
            case neighbour_type_enum::simple_neighbors:
                return simpleNeighborsRefinementsRequired;
            case neighbour_type_enum::attached_neighbors:
                return attachedNeighborsRefinementsRequired;
            case neighbour_type_enum::not_neighbors:
                return notNeighborsRefinementsRequired;
        }
    }

private:
    int updateTasks(const deviceVector<int3> &originalTasks, neighbour_type_enum neighborType);
    
    const int GaussPointsNum;
    const Mesh3D &mesh;
    const QuadratureFormula3D &qf;

    deviceVector<Point3> refinedVertices;

    int2 verticesCellsNum;
    int2 *refinedVerticesCellsNum;
    deviceVector<int3> refinedCells;
    deviceVector<double> refinedCellMeasures;
    deviceVector<int> originalCells;    //vector of indices of original triangles (with respect to the refined triangles)
    deviceVector<int> refinedCellParents;

	//tasks for refined cells (i,j and index of the original non-refined task)
    deviceVector<int3> refinedSimpleNeighborsTasks;
    deviceVector<int3> refinedAttachedNeighborsTasks;
    deviceVector<int3> refinedNotNeighborsTasks;

    //intermediate results of numerical integration (for each pair of neighbors considering refined cells)
    deviceVector<double4> d_simpleNeighborsResults;
    deviceVector<double4> d_attachedNeighborsResults;
    deviceVector<double4> d_notNeighborsResults;

    //additional buffers for tasks (in case of adaptive error control)
    deviceVector<int3> tempRefinedSimpleNeighborsTasks;
    deviceVector<int3> tempRefinedAttachedNeighborsTasks;
    deviceVector<int3> tempRefinedNotNeighborsTasks;

    //flags for successful integration calculation convergence
    deviceVector<unsigned char> simpleNeighborsIntegralsConverged;
    deviceVector<unsigned char> attachedNeighborsIntegralsConverged;
    deviceVector<unsigned char> notNeighborsIntegralsConverged;

	//buffers for mesh refinement
	deviceVector<Point3> tempVertices;
    deviceVector<int3> tempCells;
    deviceVector<double> tempCellMeasures;
    deviceVector<int> tempOriginalCells;
    deviceVector<int> cellsToBeRefined;
    deviceVector<unsigned char> cellRequiresRefinement;
    int *d_cellsToBeRefinedCount;
    int *taskCount;

	error_control_type_enum errorControlType;
    int meshRefinementLevel;

    //number of refinements required for each cell of original mesh during computation of integrals of each type
    deviceVector<unsigned char> simpleNeighborsRefinementsRequired;
    deviceVector<unsigned char> attachedNeighborsRefinementsRequired;
    deviceVector<unsigned char> notNeighborsRefinementsRequired;
};

#endif // NUMERICAL_INTEGRATOR_3D_CUH
