/*!
 * @file NumericalIntegrator3d.cuh
 * @brief NumericalIntegrator3D class which handles the process numerical integration using Gaussian quadratures
 */

#ifndef NUMERICAL_INTEGRATOR_3D_CUH
#define NUMERICAL_INTEGRATOR_3D_CUH

#include "common/constants.h"
#include "Mesh3d.cuh"
#include "QuadratureFormula3d.cuh"

/*!
 * @brief Enumerator for the error control mode
 */
enum class error_control_type_enum {
    fixed_refinement_level = 0,     //!< Fixed level of refinement for the whole mesh (memory-consuming)
    automatic_error_control = 1     //!< Automatic error control using the Rungle rule
};

/*!
 * @brief Actual evaluation of the quadrature formula
 * 
 * @param functionValues Previously evaluated values of the function in the Gauss points
 * @return Resulting value of the integral
 * 
 * Number of Gauss points and their weights are read from the constant memory
 */
__device__ double4 integrate4D(const double4 *functionValues);

/*!
 * @brief Computation of positions of Gauss points for the triangle
 * 
 * @param quadraturePoints Vector of Gauss points positions to be filled
 * @param vertices Vector of all vertices positions
 * @param triangle Indices of the vertices of the triangle
 * 
 * L-coordinates of the Gauss points are read from the constant memory. Position of \f$i\f$-th Gaussian point is calculated as
 * \f[
 *      \mathbf{r}_i = \sum\limits_{k=1}^3 Lcoord_k^{(i)}\mathbf{r}_{v_k},
 * \f]
 * where \f$\mathbf{r}_{v_k}\f$ are positions of triangle vertices.
 */
__device__ void calculateQuadraturePoints(Point3 *quadraturePoints, const Point3 *vertices, const int3 &triangle);

/*!
 * @brief Class responsible for the whole numerical integration procedure
 * 
 * The class is responsible for
 * - numerical integration itself, using Gaussian quadrature rules;
 * - forming the list of cells that require refinement;
 * - mesh refinement and creation of new tasks for the refined cells;
 * - gathering of results of integration over refined cells into original integral values.
 * 
 * Two modes are available:
 * -# Fixed level of refinement for the whole mesh (can also be zero) - memory- and time-consuming;
 * -# Automatic error control using the Runge rule.
 */
class NumericalIntegrator3D
{
public:
    /*!
     * @brief Construct a new NumericalIntegrator3D object
     * 
     * @param mesh_ Existing Mesh3D object
     * @param qf_ Quadrature formula
     * 
     * Vector of L-coordinates for Gaussian points is filled (initially the quadrature formula does not store the last coordinate)
     * and copied to the device constant memory together with weights of the Gaussian points and their number.
     * 
     * Error control mode is set to automatic error control by default. 
     */
    NumericalIntegrator3D(const Mesh3D &mesh_, const QuadratureFormula3D &qf_);
	
    /*!
     * @brief Destroy the NumericalIntegrator3D object
     * 
     * Deallocate memory for counters of tasks, refined vertices and cells, cells requiring refinement.
     */
    virtual ~NumericalIntegrator3D();

    /*!
     * @brief Set the fixed refinement level for the whole mesh
     * 
     * @param refinementLevel Specified mesh refinement level
     * 
     * Error control method is switched to fixed refinement level (default is automatic error control using the Runge rule)
     */
	void setFixedRefinementLevel(int refinementLevel = 0);

    /*!
     * @brief Necessary preparations (mesh and tasks) before integral calculation
     * 
     * @param simpleNeighborsTasks Device vector of integration tasks for simple neighbor cell pairs
     * @param attachedNeighborsTasks Device vector of integration tasks for attached neighbor cell pairs
     * @param notNeighborsTasks Device vector of integration tasks for non-neighbor cell pairs
     * 
     * -# Memory allocation for the refined mesh is performed (\f$v_0 \text{ and } t_0\f$ are the numbers of vertices and cells in the original mesh,
     * \f$n\f$ is the mesh refinement level - maximum value is used in case of adaptive error control):
     *      - number of vertices is set to \f$ v_0 + t_0 (4^n - 1) \f$;
     *      - numbers of cells and cell measures are set to \f$ t_0\cdot 4^n \f$;
     *      - indices pointing to the cells of original mesh (the size is equal to the size of vector of refined cells).
     * -# Original mesh is copied into the allocated data structures.
     * -# For each type of neighbor pairs memory is allocated for the refined integration tasks, integration results and (in case of adaptive error control)
     * flags for integration procedure convergence (according to the Runge rule) and buffer for tasks.
     * -# In case of fixed refinement level (1 or more) mesh is refined and after that tasks are created for the refined cells.
     */
	void prepareTasksAndMesh(const deviceVector<int3> &simpleNeighborsTasks, const deviceVector<int3> &attachedNeighborsTasks,	const deviceVector<int3> &notNeighborsTasks);

    /*!
     * @brief Sum up the results of integration over refined cells (for a specific type of neighbors) into the vector for original task results
     * 
     * @param results Target vector for integration results for original tasks
     * @param neighborType Type of neighbors
     * 
     * Both input and output results contain separate values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$.
     * A kernel function is called which performs summation using atomicAdd and the index of the original task for each refined one.
     */
	void gatherResults(deviceVector<double4> &results, neighbour_type_enum neighborType) const;

    /*!
     * @brief Mesh refinement (whole mesh or specific cells)
     * 
     * @param updateTasksNeighborType Type of neighbors (may be undefined)
     * 
     * The function swaps buffers for vertices, cells, cell measures and indices of original cells for the refined ones, and then call a kernel function
     * which splits the whole mesh or only selected cells. Depending on the error control mode:
     * -# Fixed refinement level: the function is called required number of times before the start of the calculation. Creation of refined tasks is not called
     * from here (as it is called in the mesh preparation procedure - once after all iterations of mesh refinement).
     * -# Adaptive error control: the function is called after each integral calculation iteration for the selected set of cells (after 1 iteration - the whole mesh).
     * Swap of buffer for refined tasks and update of tasks is performed after cells have been refined
     */
	void refineMesh(neighbour_type_enum updateTasksNeighborType = neighbour_type_enum::undefined);

    /*!
     * @brief Re-initialization of mesh data using the original mesh in Mesh3D object 
     * 
     * Mesh vertices, cells and measures are copied from the Mesh3D object, number of refined vertices and cells is reset
     * using the original numbers of vertices and cells.
     * 
     * \remark This function is used for the purpose of resetting before start of calculation for another type of neighbors
     * (the refined mesh vectors may contained fragmented mesh data from the previous integration step).
     */
    void resetMesh();

    /*!
     * @brief Fill the list of indices of cells which require further refinement, using the list of integrals which have not converged
     * 
     * @param restTasks Indices of integration tasks which have not yet converged
     * @param tasks List of all integration task on current iteration
     * @param neighborType Type of neighbors
     * @return int Number of cells which require refinement
     * 
     * The following kernels are called:
     * -# Setting to 'true' the flag of cells which are control panels of at least one integration task which has not converged.
     * -# Extracting indices of the flags set to 'true'.
     * -# Increasing the number of required requirements for these cells by 1 (used for output to VTK).
     */
    int determineCellsToBeRefined(const deviceVector<int> &restTasks, const deviceVector<int3> *tasks, neighbour_type_enum neighborType);

    /*!
     * @brief Get the number of Gauss Points
     * 
     * @return int Number of Gauss points
     * 
     * Used for passing the number of Gaussian points to kernel functions
     */
	int getGaussPointsNumber() const {
		return GaussPointsNum;
	}

    /*!
     * @brief Get the order of the specified quadrature formula
     * 
     * @return int Order of the selected quarature formula
     */
    int getQuadratureFormulaOrder() const {
        return qf.order;
    }

    /*!
     * @brief Get the type of error control
     * 
     * @return error_control_type_enum Current type of error control mode
     */
	error_control_type_enum getErrorControlType() const {
		return errorControlType;
	}

    /*!
     * @brief Get the vector of integration tasks for refined cells (for a specific type of neighbor cells)
     * 
     * @param neighborType Type of neighbors
     * @return const deviceVector<int3>* Pointer to device vector of tasks
     */
    const deviceVector<int3> *getRefinedTasks(neighbour_type_enum neighborType) const {
        switch(neighborType)
        {
        case neighbour_type_enum::simple_neighbors:
            return &refinedSimpleNeighborsTasks;
        case neighbour_type_enum::attached_neighbors:
            return &refinedAttachedNeighborsTasks;
        case neighbour_type_enum::not_neighbors:
            return &refinedNotNeighborsTasks;
        default:
            return nullptr;
        }
    }

    /*!
     * @brief Get the vector of results of integration for refined cells (for a specific type of neighbor cells)
     * 
     * @param neighborType Type of neighbors
     * @return const deviceVector<double4>* Pointer to device vector of integration results
     * 
     * Elements of this vector are later summed up in order to fill the vector of integral values for original integration tasks
     */
    const deviceVector<double4> *getResults(neighbour_type_enum neighborType) const {
        switch(neighborType)
        {
        case neighbour_type_enum::simple_neighbors:
            return &d_simpleNeighborsResults;
        case neighbour_type_enum::attached_neighbors:
            return &d_attachedNeighborsResults;
        case neighbour_type_enum::not_neighbors:
            return &d_notNeighborsResults;
        default:
            return nullptr;
        }
    }

    /*!
     * @brief Get the vector of refined vertices
     * 
     * @return const auto& Device vector of coordinates of refined vertices
     * 
     * The vector is used for passing as an argument to kernel functions and for export to VTK. The list contains
     * vertices from the previous refinement iteration after which new vertices are added.
     * 
     * The vertices may not cover the whole mesh if adaptive mesh refinement procedure is used (only specific cells are refined
     * and their vertices are added)
     */
    const auto &getRefinedVertices() const {
        return refinedVertices;
    }

    /*!
     * @brief Get the vector of refined cells
     * 
     * @return const auto& Device vector of vertex indices for the refined cells
     * 
     * The vector is used for passing as an argument to kernel functions and for export to VTK. The list is filled from the beginning
     * on each refinement iteration and does not contain cells from the previous iteration.
     * 
     * The cells may not cover the whole mesh if adaptive mesh refinement procedure is used (only specific cells are refined
     * and new cells for them are added)
     */
    const auto &getRefinedCells() const {
        return refinedCells;
    }

    /*!
     * @brief Get the vector of measures (areas) of refined cells
     * 
     * @return const auto& Device vector of measures (areas) for the refined cells
     * 
     * The vector is used for passing as an argument to kernel functions.
     * 
     * The values for cells may not cover the whole mesh if adaptive mesh refinement procedure is used (only specific cells are refined
     * and measures for new cells are added)
     */
	const auto &getRefinedCellMeasures() const {
        return refinedCellMeasures;
    }

    /*!
     * @brief Get the vector of indices of cells which are to be refined
     * 
     * @return auto& Device vector of indices (in the whole current list of cells) of cells which are to be refined
     */
    auto &getCellsToBeRefined() const {
        return cellsToBeRefined;
    }

    /*!
     * @brief Get the vector of flags which show whether the integration of the task has already converged (for a specific type of neighbor cells)
     * 
     * @param neighborType Type of neighbors
     * @return const deviceVector<unsigned char>* Pointer to device vector of flags for convergence of integration tasks
     * 
     * The values of flags are either 'true' or 'false'. Flags are assigned during the procedure of comparing integration results
     * between the current and previous iterations. Later new tasks are created only for the original tasks whose flag is still 'false'.
     */
    deviceVector<unsigned char> *getIntegralsConverged(neighbour_type_enum neighborType){
        switch(neighborType){
            case neighbour_type_enum::simple_neighbors:
                return &simpleNeighborsIntegralsConverged;
            case neighbour_type_enum::attached_neighbors:
                return &attachedNeighborsIntegralsConverged;
            case neighbour_type_enum::not_neighbors:
                return &notNeighborsIntegralsConverged;
            default:
                return nullptr;
        }
    }

    /*!
     * @brief Get the vector with number of required refinements for each cell (for a specific type of neighbor cells)
     * 
     * @param neighborType Type of neighbors
     * @return const deviceVector<unsigned char>* Pointer to device vector with number of required refinements for each cell 
     * 
     * Data of this vector is used for output to the VTK file. Unsigned char data type is used due to the number of refinements
     * being usually not higher than 10.
     */
    deviceVector<unsigned char> *getRefinementsRequired(neighbour_type_enum neighborType){
        switch(neighborType){
            case neighbour_type_enum::simple_neighbors:
                return &simpleNeighborsRefinementsRequired;
            case neighbour_type_enum::attached_neighbors:
                return &attachedNeighborsRefinementsRequired;
            case neighbour_type_enum::not_neighbors:
                return &notNeighborsRefinementsRequired;
            default:
                return nullptr;
        }
    }

private:
    /*!
     * @brief Create new integration tasks for the refined cells, based on the the existing list of tasks for parent cells  
     * 
     * @param originalTasks Vector of tasks to be used to create new tasks for refined cells
     * @param neighborType Type of neighbors
     * @return int New number of tasks (for the refined cells)
     * 
     * Two (almost) identical calls of a kernel function are performed. First is needed to count the number of tasks to be created.
     * During the second run the tasks are actually created.
     * 
     * Situation is a bit different depending on the error control mode:
     * - in case of fixed mesh refinement level tasks creation is perfomed only after the mesh has been refined the desired number of times.
     * Then the memory for refined tasks is allocated and new tasks are created based on the original ones;
     * - in case of adaptive mesh control tasks are created after every integration procedure.
     * In this case vectors of refined integration tasks and results are resized accordingly and new tasks are created using the tasks from the previous iteration. 
     */
    int updateTasks(const deviceVector<int3> &originalTasks, neighbour_type_enum neighborType);
    
    const int GaussPointsNum;                           //!< Number of Gaussian points in the quadrature formula

    /*!
     * @brief A Mesh3D object with the original surface mesh
     * 
     * Vertices, cells and measures are used from the original mesh
     */
    const Mesh3D &mesh;

    /*!
     * @brief A quadrature formula selected for numerical integration
     * 
     * Data from the quadrature formula (which is a static object) is copied to the device constant memory
     */
    const QuadratureFormula3D &qf;

    /*!
     * @brief Device vector of coordinates of refined vertices
     * 
     * The vertices may not cover the whole mesh if adaptive mesh refinement procedure is used (only specific cells are refined
     * and their vertices are added)
     */
    deviceVector<Point3> refinedVertices;

    /*!
     * @brief Current numbers of vertices and cells in the refined mesh stored on the host
     * 
     * The value is initially assigned using the number in original mesh, then it is updated by the values from the device
     * which are obtained in the mesh refinement procedure. 2 numbers are stored as 1 int2 value (.x refers to vertices, .y - to cells)
     */
    int2 verticesCellsNum;

    /*!
     * @brief Current numbers of vertices and cells in the refined mesh stored on the device
     * 
     * The value is initialized by the value in the host memory, then it is updated in the mesh refinement procedure.
     * 2 numbers are stored as 1 int2 value (.x refers to vertices, .y - to cells)
     */
    int2 *refinedVerticesCellsNum = nullptr;

    /*!
     * @brief Device vector of vertex indices for the refined cells
     * 
     * The cells may not cover the whole mesh if adaptive mesh refinement procedure is used (only specific cells are refined
     * and new cells for them are added)
     */
    deviceVector<int3> refinedCells;

    /*!
     * @brief Device vector of measures (areas) for the refined cells
     * 
     * The values for cells may not cover the whole mesh if adaptive mesh refinement procedure is used (only specific cells are refined
     * and measures for new cells are added)
     */
    deviceVector<double> refinedCellMeasures;

    /*!
     * @brief Vector of indices of original triangles (belong to the refined triangles)
     * 
     * The index points to a cell in the original mesh. 
     * 
     * Initially the list is filled with \f$0,1,\ldots,n-1\f$, and later during the refinements the index is handed over
     * from parent triangle to child ones without any changes.
     */
    deviceVector<int> originalCells;

    /*!
     * @brief Vector of indices of direct parent cells of the refined cells
     * 
     * The index points to a cell in the list for the previous iteration.
     * 
     * The values are used in the procedure of task creation after mesh refinement in order to decide whether a new task
     * should be created for an old one or not.
     */
    deviceVector<int> refinedCellParents;

    /*!
     * @brief Vector of tasks for integration over refined simple neighbor pairs
     * 
     * The tasks has the form \f$(i,j,k)\f$ where \f$i,j\f$ correspond to the indices of refined cells
     * and \f$k\f$ is the index of original non-refined task. The vector of tasks is obtained from the previous
     * vector tasks (original tasks or already refined list) after cells are refined
     */
    deviceVector<int3> refinedSimpleNeighborsTasks;

    /*!
     * @brief Vector of tasks for integration over refined attached neighbor pairs
     * 
     * The tasks has the form \f$(i,j,k)\f$ where \f$i,j\f$ correspond to the indices of refined cells
     * and \f$k\f$ is the index of original non-refined task. The vector of tasks is obtained from the previous
     * vector tasks (original tasks or already refined list) after cells are refined
     */
    deviceVector<int3> refinedAttachedNeighborsTasks;
    
    /*!
     * @brief Vector of tasks for integration over refined non-neighbor pairs
     * 
     * The tasks has the form \f$(i,j,k)\f$ where \f$i,j\f$ correspond to the indices of refined cells
     * and \f$k\f$ is the index of original non-refined task. The vector of tasks is obtained from the previous
     * vector tasks (original tasks or already refined list) after cells are refined
     */
    deviceVector<int3> refinedNotNeighborsTasks;

    /*!
     * @brief Results of integration tasks over refined simple neighbor pairs
     * 
     * Values in this list are later summed up in order to obtain values of integrals for original tasks
     * (using indices for correspondence of original and refined tasks). Each entry stores separately values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$
     */
    deviceVector<double4> d_simpleNeighborsResults;
    
    /*!
     * @brief Results of integration tasks over refined attached neighbor pairs
     * 
     * Values in this list are later summed up in order to obtain values of integrals for original tasks
     * (using indices for correspondence of original and refined tasks). Each entry stores separately values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$
     */
    deviceVector<double4> d_attachedNeighborsResults;

    /*!
     * @brief Results of integration tasks over refined non-neighbor pairs
     * 
     * Values in this list are later summed up in order to obtain values of integrals for original tasks
     * (using indices for correspondence of original and refined tasks). Each entry stores separately values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$
     */
    deviceVector<double4> d_notNeighborsResults;

    /*!
     * @brief Additional buffer for tasks for integration over refined simple neighbor pairs
     * 
     * Used in case of adaptive error control procedure, this buffer is swapped with the main list of refined tasks alternatively,
     * at each integration iteration.
     */
    deviceVector<int3> tempRefinedSimpleNeighborsTasks;

    /*!
     * @brief Additional buffer for tasks for integration over refined attached neighbor pairs
     * 
     * Used in case of adaptive error control procedure, this buffer is swapped with the main list of refined tasks alternatively,
     * at each integration iteration.
     */

    deviceVector<int3> tempRefinedAttachedNeighborsTasks;

    /*!
     * @brief Additional buffer for tasks for integration over refined non-neighbor pairs
     * 
     * Used in case of adaptive error control procedure, this buffer is swapped with the main list of refined tasks alternatively,
     * at each integration iteration.
     */
    deviceVector<int3> tempRefinedNotNeighborsTasks;

    /*!
     * @brief Vector of flags which show whether the integration of the task has already converged (for the original, non-refined simple neighbor pairs)
     * 
     * The values of flags are either 'true' or 'false'. Flags are assigned during the procedure of comparing integration results
     * between the current and previous iterations. Later new tasks are created only for the original tasks whose flag is still 'false'.
     */
    deviceVector<unsigned char> simpleNeighborsIntegralsConverged;

    /*!
     * @brief Vector of flags which show whether the integration of the task has already converged (for the original, non-refined attached neighbor pairs)
     * 
     * The values of flags are either 'true' or 'false'. Flags are assigned during the procedure of comparing integration results
     * between the current and previous iterations. Later new tasks are created only for the original tasks whose flag is still 'false'.
     */
    deviceVector<unsigned char> attachedNeighborsIntegralsConverged;

    /*!
     * @brief Vector of flags which show whether the integration of the task has already converged (for the original, non-refined non-neighbor pairs)
     * 
     * The values of flags are either 'true' or 'false'. Flags are assigned during the procedure of comparing integration results
     * between the current and previous iterations. Later new tasks are created only for the original tasks whose flag is still 'false'.
     */
    deviceVector<unsigned char> notNeighborsIntegralsConverged;

    /*!
     * @brief Temporary buffer for coordinates of refined mesh vertices
     * 
     * The vector is used during the refinement procedure in case of fixed mesh refinement level, or is swapped alternatively
     * with the main vector of coordinates of refined mesh vertices in case of adaptive error control procedure
     */
	deviceVector<Point3> tempVertices;

    /*!
     * @brief Temporary buffer for vertex indices of refined mesh cells
     * 
     * The vector is used during the refinement procedure in case of fixed mesh refinement level, or is swapped alternatively
     * with the main vector of vertex indices of refined mesh cells in case of adaptive error control procedure
     */
    deviceVector<int3> tempCells;

    /*!
     * @brief Temporary buffer for vector of measures (areas) of refined cells
     * 
     * The vector is used during the refinement procedure in case of fixed mesh refinement level, or is swapped alternatively
     * with the main vector of vector of measures (areas) of refined cells in case of adaptive error control procedure
     */
    deviceVector<double> tempCellMeasures;

    /*!
     * @brief Temporary buffer for vector of indices of original triangles for refined cells
     * 
     * The vector is used during the refinement procedure in case of fixed mesh refinement level, or is swapped alternatively
     * with the main vector of vector of indices of original triangles for refined cells in case of adaptive error control procedure
     */
    deviceVector<int> tempOriginalCells;

    /*!
     * @brief Vector of indices of cells which need to be refined for further calculation of integrals over them
     * 
     * Indices point to positions in the list of refined cells
     */
    deviceVector<int> cellsToBeRefined;

    /*!
     * @brief Intermediate vector of flags indicating that a cell (from the original list) requires refinement
     * 
     * The values are obtained by analyzing the list of tasks which are still left for integration, the next kernel function
     * extracts indices of such cells, which are then refined
     */
    deviceVector<unsigned char> cellRequiresRefinement;
    
    int *d_cellsToBeRefinedCount = nullptr;             //!< Device counter of a number of cells which need to be refined
    
    /*!
     * @brief Device counter of a number of refined tasks
     * 
     * The value is accumulated during execution of a kernel function while analyzes the existing tasks and refined cells.
     * The value serves the purpose of determining the size (and allocating/re-allocating memory if necessary)
     * for vectors of refined tasks and integration results.
     */
    int *taskCount = nullptr;

    /*!
     * @brief Specified mode of error control
     * 
     * Default value is automatic error control procedure using the Runge rule
     */
	error_control_type_enum errorControlType;

    /*!
     * @brief Specified level of mesh refinement
     * 
     * The value is only used if the mode of full mesh refinement is selected.
     * Mesh refinement level can be set to 0, then integrals are calculated on the original mesh.
     */
    int meshRefinementLevel;

    /*!
     * @brief Vector with number of required refinements for each cell when integrating over simple neighbor pairs
     * 
     * The cell is consider to be \f$i\f$-th cell in pair \f$(i,j)\f$ (control panel).
     * Unsigned char data type is used due to the number of refinements being usually not higher than 10.
     */
    deviceVector<unsigned char> simpleNeighborsRefinementsRequired;
    
    /*!
     * @brief Vector with number of required refinements for each cell when integrating over attached neighbor pairs
     * 
     * The cell is consider to be \f$i\f$-th cell in pair \f$(i,j)\f$ (control panel).
     * Unsigned char data type is used due to the number of refinements being usually not higher than 10.
     */
    deviceVector<unsigned char> attachedNeighborsRefinementsRequired;

    /*!
     * @brief Vector with number of required refinements for each cell when integrating over non-neighbor pairs
     * 
     * The cell is consider to be \f$i\f$-th cell in pair \f$(i,j)\f$ (control panel).
     * Unsigned char data type is used due to the number of refinements being usually not higher than 10.
     */
    deviceVector<unsigned char> notNeighborsRefinementsRequired;
};

#endif // NUMERICAL_INTEGRATOR_3D_CUH
