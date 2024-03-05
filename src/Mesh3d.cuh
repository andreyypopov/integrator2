#ifndef MESH3D_CUH
#define MESH3D_CUH

#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <string>
#include <vector>
#include <array>

enum class neighbour_type_enum {
    simple_neighbors = 0,
    attached_neighbors = 1,
    not_neighbors = 2,
    undefined = -1
};

class Mesh3D
{
public:
    virtual ~Mesh3D();

    bool loadMeshFromFile(const std::string &filename, double scale = 1.0);

    void prepareMesh();

    const auto &getSimpleNeighbors() const {
        return simpleNeighbors;
    }

    const auto &getAttachedNeighbors() const {
        return attachedNeighbors;
    }

    const auto &getNotNeighbors() const {
        return notNeighbors;
    }

    const auto &getVertices() const {
        return vertices;
    }

    const auto &getCells() const {
        return cells;
    }

    const auto &getCellNormals() const {
        return cellNormals;
    }

    const auto &getCellMeasures() const {
        return cellMeasures;
    }

private:
    void calculateNormals();

    void calculateCenters();

    void calculateMeasures();

    void fillNeightborsLists();

    deviceVector<Point3> vertices;
        
    deviceVector<int3> cells;
    deviceVector<Point3> cellNormals;
    deviceVector<Point3> cellCenters;
    deviceVector<double> cellMeasures;
    
    int *d_simpleNeighborsNum = nullptr;
    int *d_attachedNeighborsNum = nullptr;
    int *d_notNeighborsNum = nullptr;
    deviceVector<int3> simpleNeighbors;
    deviceVector<int3> attachedNeighbors;
    deviceVector<int3> notNeighbors;
};

void exportMeshToObj(const std::string &filename, const std::vector<Point3> &vertices, const std::vector<int3> &cells);

void exportMeshToVtk(const std::string &filename, const std::vector<Point3> &vertices, const std::vector<int3> &cells,
        const std::array<std::vector<unsigned char>, 3> &refinementsRequired);

std::string neighborTypeString(neighbour_type_enum neighborType);

#endif // MESH3D_CUH