#ifndef MESH3D_CUH
#define MESH3D_CUH

#include "common/cuda_math.cuh"
#include "common/device_vector.cuh"

#include <string>
#include <vector>
#include <array>

/*!
* \brief Enumerator which describes size of intersection between 2 neighboring cells
*/
enum class neighbour_type_enum {
    simple_neighbors = 0,   //!< 1 common vertex for triangles
    attached_neighbors = 1, //!< 2 common vertices for triangles (a common edge)
    not_neighbors = 2,      //!< No common vertices for triangles (empty intersection)
    undefined = -1          //!< Unspecified dummy value
};

/*!
 * \brief Class for storage of mesh information
 *
 * Main functionality:
 * - import of mesh data (vertices and cells) from the DAT file;
 * - mesh processing (calculation of normals, measures, centers);
 * - determining cell neighbors.
 *
 * \remark <b>All</b> the data is stored in the GPU memory
*/
class Mesh3D
{
public:
    /*!
     * \brief Destructor
     *
     * Free GPU memory for counters of neighbors of different types    
     */
    virtual ~Mesh3D();

    /*!
     * @brief Import mesh from the DAT file
     * 
     * @param filename Name of the .dat file
     * @param scale Scaling parameter for vertex coordinates
     * @return true Mesh successfully loaded and transferred to the GPU memory
     * @return false Error while loading the mesh
     */
    bool loadMeshFromFile(const std::string &filename, double scale = 1.0);

    /*!
     * @brief Unified function with calls for all preparatory functions for the mesh
     * 
     * Includes calculation of cell normals, centers and measures and determining types of neighbors for all cell pairs.
     * All procedures are carried out on GPU.
     */
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

    deviceVector<Point3> vertices;              //!< Vector of vertex coordinates
        
    deviceVector<int3> cells;                   //!< Vector of indices of vertices describing each cell
    deviceVector<Point3> cellNormals;           //!< Coordinates of cell normal vectors
    deviceVector<Point3> cellCenters;           //!< Coordinates of cell centers
    deviceVector<double> cellMeasures;          //!< Cell measures (areas in 3D)
    
    int *d_simpleNeighborsNum = nullptr;        //!< Counter of simple neighbor pairs (only (i, j) where i<j are considered)
    int *d_attachedNeighborsNum = nullptr;      //!< Counter of attached neighbor pairs (only (i, j) where i<j are considered)
    int *d_notNeighborsNum = nullptr;           //!< Counter of non-neighbor pairs (only (i, j) where i<j are considered)
    deviceVector<int3> simpleNeighbors;         //!< Vector of simple neighbor pairs (3rd coordinate contain the index for the purpose of further refinement)
    deviceVector<int3> attachedNeighbors;       //!< Vector of attached neighbor pairs (3rd coordinate contain the index for the purpose of further refinement)
    deviceVector<int3> notNeighbors;            //!< Vector of non-neighbor pairs (3rd coordinate contain the index for the purpose of further refinement)
};

void exportMeshToObj(const std::string &filename, const std::vector<Point3> &vertices, const std::vector<int3> &cells);

void exportMeshToVtk(const std::string &filename, const std::vector<Point3> &vertices, const std::vector<int3> &cells,
        const std::array<std::vector<unsigned char>, 3> &refinementsRequired);

std::string neighborTypeString(neighbour_type_enum neighborType);

#endif // MESH3D_CUH