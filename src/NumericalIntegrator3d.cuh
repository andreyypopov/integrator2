#ifndef NUMERICAL_INTEGRATOR_3D_CUH
#define NUMERICAL_INTEGRATOR_3D_CUH

#include <vector>

#include "common/constants.h"
#include "Mesh3d.cuh"

struct QuadratureFormula3D
{
    const std::vector<double2> coordinates;
    const std::vector<double> weights;

    const int order;
};

/// Quadrature formula with 1 Gauss point (2nd order)
const QuadratureFormula3D qf3D1{
{
	{ 0.3333333333333333, 0.3333333333333333 },	
},
{
	1.0	
},
	2
};

/// Quadrature formula with 3 Gauss points (2nd order)
const QuadratureFormula3D qf3D3{
{
	{ 0.1666666666666667, 0.1666666666666667 },
	{ 0.6666666666666667, 0.1666666666666667 },
	{ 0.1666666666666667, 0.6666666666666667 },
},
{
	0.3333333333333333,
	0.3333333333333333,
	0.3333333333333333
},
	2
};

/// Quadrature formula with 4 Gauss points (3rd order)
const QuadratureFormula3D qf3D4{
{
	{ 0.3333333333333333, 0.3333333333333333 },
	{ 0.6, 0.2 },
	{ 0.2, 0.6 },
	{ 0.2, 0.2 }
},
{
	-0.5625,
	0.5208333333333333,
	0.5208333333333333,
	0.5208333333333333
},
	3
};

/// Quadrature formula with 6 Gauss points (4th order)
const QuadratureFormula3D qf3D6{
{
	{ 0.816847572980459, 0.091576213509771 },
	{ 0.091576213509771, 0.816847572980459 },
	{ 0.091576213509771, 0.091576213509771 },
	{ 0.445948490915965, 0.108103018168070 },
	{ 0.108103018168070, 0.445948490915965 },
	{ 0.445948490915965, 0.445948490915965 }	
},
{
	0.109951743655322,
	0.109951743655322,
	0.109951743655322,
	0.223381589678011,
	0.223381589678011,
	0.223381589678011
},
	4
};

/// Quadrature formula with 7 Gauss points (5th order)
const QuadratureFormula3D qf3D7{
{
	{ 0.3333333333333333, 0.3333333333333333 },
	{ 0.101286507323456, 0.797426985353087 },
	{ 0.797426985353087, 0.101286507323456 },
	{ 0.101286507323456, 0.101286507323456 },
	{ 0.470142064105115, 0.059715871789770 },
	{ 0.059715871789770, 0.470142064105115 },
	{ 0.470142064105115, 0.470142064105115 }
},
{
	0.225,
	0.125939180544827,
	0.125939180544827,
	0.125939180544827,
	0.132394152788506,
	0.132394152788506,
	0.132394152788506
},
	5
};

/// Quadrature formula with 9 Gauss points (5th order)
const QuadratureFormula3D qf3D9{
{
	{ 0.437525248383384, 0.124949503233232 },
	{ 0.124949503233232, 0.437525248383384 },
	{ 0.437525248383384, 0.437525248383384 },
	{ 0.797112651860071, 0.165409927389841 },
	{ 0.165409927389841, 0.797112651860071 },
	{ 0.797112651860071, 0.037477420750088 },
	{ 0.037477420750088, 0.797112651860071 },
	{ 0.165409927389841, 0.037477420750088 },
	{ 0.037477420750088, 0.165409927389841 }
},
{
	0.205950504760887,
	0.205950504760887,
	0.205950504760887,
	0.063691414286223,
	0.063691414286223,
	0.063691414286223,
	0.063691414286223,
	0.063691414286223,
	0.063691414286223,	
},
	5
};

/// Quadrature formula with 12 Gauss points (6th order)
const QuadratureFormula3D qf3D12{
{
	{ 0.873821971016996, 0.063089014491502 },
	{ 0.063089014491502, 0.873821971016996 },
	{ 0.063089014491502, 0.063089014491502 },
	{ 0.501426509658179, 0.249286745170910 },
	{ 0.249286745170910, 0.501426509658179 },
	{ 0.249286745170910, 0.249286745170910 },
	{ 0.636502499121399, 0.310352451033785 },
	{ 0.310352451033785, 0.636502499121399 },
	{ 0.636502499121399, 0.053145049844816 },
	{ 0.053145049844816, 0.636502499121399 },
	{ 0.310352451033785, 0.053145049844816 }, 
	{ 0.053145049844816, 0.310352451033785 }	
},
{
	0.050844906370207,
	0.050844906370207,
	0.050844906370207,
	0.116786275726379,
	0.116786275726379,
	0.116786275726379,
	0.082851075618374,
	0.082851075618374,
	0.082851075618374,
	0.082851075618374,
	0.082851075618374,
	0.082851075618374	
},
	6
};

/// Quadrature formula with 13 Gauss points (7th order)
const QuadratureFormula3D qf3D13{
{
	{ 0.333333333333333, 0.333333333333333 },
	{ 0.479308067841923, 0.260345966079038 },
	{ 0.260345966079038, 0.479308067841923 },
	{ 0.260345966079038, 0.260345966079038 },
	{ 0.869739794195598, 0.065130102902216 },
	{ 0.065130102902216, 0.869739794195598 },
	{ 0.065130102902216, 0.065130102902216 },
	{ 0.638444188569809, 0.312865496004875 },
	{ 0.312865496004875, 0.638444188569809 },
	{ 0.638444188569809, 0.048690315425316 },
	{ 0.048690315425316, 0.638444188569809 },
	{ 0.312865496004875, 0.048690315425316 },
	{ 0.048690315425316, 0.312865496004875 }
}, 
{
   -0.149570044467670,
	0.175615257433204,
	0.175615257433204,
	0.175615257433204,
	0.053347235608839,
	0.053347235608839,
	0.053347235608839,
	0.077113760890257,
	0.077113760890257,
	0.077113760890257,
	0.077113760890257,
	0.077113760890257,
	0.077113760890257
},
	7
};

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
