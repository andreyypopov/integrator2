#ifndef MESH3D_CUH
#define MESH3D_CUH

#include "common/cuda_math.cuh"

#include <string>

enum class neighbour_type_enum {
    simple_neighbors = 0,
    attached_neighbors = 1,
    not_neighbors = 2
};

class Mesh3D
{
public:
    virtual ~Mesh3D();

    bool loadMeshFromFile(const std::string &filename, double scale = 1.0);

    void prepareMesh();

    int getSimpleNeighborsNum() const {
        return simpleNeighborsNum;
    }

    const int2 *getSimpleNeighbors() const {
        return simpleNeighbors;
    }

    int getAttachedNeighborsNum() const {
        return attachedNeighborsNum;
    }

    const int2 *getAttachedNeighbors() const {
        return attachedNeighbors;
    }

    int getNotNeighborsNum() const {
        return notNeighborsNum;
    }

    const int2 *getNotNeighbors() const {
        return notNeighbors;
    }
private:
    void calculateNormals();

    void calculateCenters();

    void calculateMeasures();

    void fillNeightborsLists();

    int verticesNum = 0;
    Point3 *vertices;
        
    int cellsNum = 0;
    int3 *cells;
    Point3 *cellNormals;
    Point3 *cellCenters;
    double *cellMeasures;
    
    int quadraturePointsNum = 0;
    Point3 *quadraturePoints;

    int simpleNeighborsNum = 0;
    int attachedNeighborsNum = 0;
    int notNeighborsNum = 0;
    int *d_simpleNeighborsNum;
    int *d_attachedNeighborsNum;
    int *d_notNeighborsNum;
    int2 *simpleNeighbors;
    int2 *attachedNeighbors;
    int2 *notNeighbors;
};

#endif // MESH3D_CUH