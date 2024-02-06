#ifndef EVALUATORJ3DK_CUH
#define EVALUATORJ3DK_CUH

#include "evaluator3d.cuh"

__device__ double4 thetaPsi(const Point3 &pt, const Point3 *vertices, const int3 &triangle);

__device__ double4 singularPartAttached(const Point3 &pt, int i, int j, const Point3 *vertices, const int3 *cells);

__device__ double4 singularPartSimple(const Point3 &pt, int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures);

__device__ double4 integrateSingularPartAttached(int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures);

__device__ double4 integrateSingularPartSimple(int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures);

__device__ int2 shiftsForSimpleNeighbors(const int3 &triangle1, const int3 &triangle2);

__device__ int2 shiftsForAttachedNeighbors(const int3 &triangle1, const int3 &triangle2);

__device__ int3 rotateLeft(const int3 &triangle, int shift);

class EvaluatorJ3DK : public Evaluator3D
{
public:
    EvaluatorJ3DK(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_)
        : Evaluator3D(mesh_, numIntegrator_){ };

    virtual void integrateOverSimpleNeighbors() override;
    virtual void integrateOverAttachedNeighbors() override;
    virtual void integrateOverNotNeighbors() override;
};

#endif // EVALUATORJ3DK_CUH
