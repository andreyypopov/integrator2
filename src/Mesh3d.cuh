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
 * \remark Neighbor pairs are stored as *(i,j,k)*, where *i* and *j* are indices of cells and *k*
 * is the index of the pair itself (it is used further in the integration process in case of cell refinement)
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

    /*!
     * @brief Get the simple neighbors vector
     * 
     * @return const auto& Vector of simple neighbors
     */
    const auto &getSimpleNeighbors() const {
        return simpleNeighbors;
    }

    /*!
     * @brief Get the attached neighbors vector
     * 
     * @return const auto& Vector of attached neighbors
     */
    const auto &getAttachedNeighbors() const {
        return attachedNeighbors;
    }

    /*!
     * @brief Get the non-neighbors vector
     * 
     * @return const auto& Vector of non-neighbors
     */
    const auto &getNotNeighbors() const {
        return notNeighbors;
    }

    /*!
     * @brief Get the vertices coordinates vector
     * 
     * @return const auto& Vector of vertices coordinates
     */
    const auto &getVertices() const {
        return vertices;
    }

    /*!
     * @brief Get the indices of cell vertices
     * 
     * @return const auto& Vector of indices of cell vertices
     */
    const auto &getCells() const {
        return cells;
    }

    /*!
     * @brief Get the cell normals vector
     * 
     * @return const auto& Vector of cell normals
     */
    const auto &getCellNormals() const {
        return cellNormals;
    }

    /*!
     * @brief Get the cell measures vector
     * 
     * @return const auto& Vector of cell measures
     */
    const auto &getCellMeasures() const {
        return cellMeasures;
    }

private:
    /*!
     * @brief Calculation of cell normals
     *
     * A kernel function is called which computes each cell's normal as a cross-product of its 2 edges with a common vertex 
     */
    void calculateNormals();

    /*!
     * @brief Calculation of cell centers
     * 
     * A kernel function is called which computes each cell's center as mean value of its vertex coordinates.
     */
    void calculateCenters();

    /*!
     * @brief Calculation of cell measures
     * 
     * A kernel function is called which computes each cell's measure as 1/2 of magnitude of cross-product of its 2 edges with a common vertex
     */
    void calculateMeasures();

    /*!
     * @brief Fill the lists of neighbors of different types
     * 
     * Memory is allocated for counters and vectors of different types of neighbors using the following rules:
     * -# only pairs *(i,j)* where *i<j* are considered;
     * -# for simple neighbors a specific constant is used for maximum number of simple neighbors for a single cell;
     * -# for attached neighbors - exactly 3 (across each edge);
     * -# for non-neighbors simply all-to-all pairs.
     * 
     * A kernel function is called which compares vertex indices of each pair of triangles and adds this pair to a corresponding list. 
     * The obtained lists are then used as integration tasks.
     */
    void fillNeightborsLists();

    deviceVector<Point3> vertices;              //!< Vector of vertices coordinates
        
    deviceVector<int3> cells;                   //!< Vector of indices of vertices describing each cell
    deviceVector<Point3> cellNormals;           //!< Coordinates of cell normal vectors
    deviceVector<Point3> cellCenters;           //!< Coordinates of cell centers
    deviceVector<double> cellMeasures;          //!< Cell measures (areas in 3D)
    
    int *d_simpleNeighborsNum = nullptr;        //!< Counter of simple neighbor pairs (only (*i, j*) where *i<j* are considered)
    int *d_attachedNeighborsNum = nullptr;      //!< Counter of attached neighbor pairs (only (*i, j*) where *i<j* are considered)
    int *d_notNeighborsNum = nullptr;           //!< Counter of non-neighbor pairs (only (*i, j*) where *i<j* are considered)
    deviceVector<int3> simpleNeighbors;         //!< Vector of simple neighbor pairs (3rd coordinate contain the index for the purpose of further refinement)
    deviceVector<int3> attachedNeighbors;       //!< Vector of attached neighbor pairs (3rd coordinate contain the index for the purpose of further refinement)
    deviceVector<int3> notNeighbors;            //!< Vector of non-neighbor pairs (3rd coordinate contain the index for the purpose of further refinement)
};

/*!
 * @brief Export mesh to OBJ file
 * 
 * @param filename Resulting file name
 * @param vertices (Host) vector of vertices coordinates
 * @param cells (Host) vector of indices of cell vertices
 * 
 * Vertices coordinates and indices of vertices of cells (base-1) are exported
 */
void exportMeshToObj(const std::string &filename, const std::vector<Point3> &vertices, const std::vector<int3> &cells);

/*!
 * @brief Export mesh (and optionally mesh refinement information) to VTK file
 * 
 * @param filename Resulting file name
 * @param vertices (Host) vector of vertices coordinates
 * @param cells (Host) vector of indices of cell vertices
 * @param refinementsRequired 3 vectors with numbers of necessary refinement levels for cells (obtained after integration), can empty
 * 
 * Data is exported to .vtp (polygonal data) XML-type file.  
 */
void exportMeshToVtk(const std::string &filename, const std::vector<Point3> &vertices, const std::vector<int3> &cells,
        const std::array<std::vector<unsigned char>, 3> &refinementsRequired);

/*!
 * @brief String for a specific neighbor type (for message generation)
 * 
 * @param neighborType Enumeration value of neighbor type
 * @return std::string Resulting string with neighbor type name
 */
std::string neighborTypeString(neighbour_type_enum neighborType);

#endif // MESH3D_CUH