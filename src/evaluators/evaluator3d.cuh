#ifndef EVALUATOR3D_CUH
#define EVALUATOR3D_CUH

#include <vector>

#include "../common/gpu_timer.cuh"
#include "../NumericalIntegrator3d.cuh"
#include "../Mesh3d.cuh"

/*!
 * @brief Enumerator which describes the format of the output file with results of integration
 * 
 */
enum class output_format_enum {
    plainText = 1,      //!< Integration results are exported in a structured plain text file
    csv = 2             //!< Integration results are exported in a table format CSV file
};

/*!
 * @brief Parent class for evaluation of integrals in 3D
 * 
 * Contains main integration procedure, preceded by preparation of necessary data vectors, which then calls specific
 * integration functions defined in the child classes. Results can be exported to a file.
 */
class Evaluator3D
{
public:
    /*!
     * @brief Construct a new Evaluator3D object
     * 
     * @param mesh_ Mesh3D object with a loaded surface mesh
     * @param numIntegrator_ NumericalIntegrator3D object with a previously initialized quadrature formula object
     * 
     * Value of \f$2^p\f$, where \f$p\f$ is the order of quadrature formula,
     * is copied to the constant memory (used only in case of adaptive error control procedure).
     */
    Evaluator3D(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_);
    
    /*!
     * @brief Destroy the Evaluator3D object
     * 
     * Device counter of remaining tasks is deallocated.
     */
    virtual ~Evaluator3D();

    virtual void integrateOverSimpleNeighbors() = 0;        //!< Pure virtual function for integration over simple neighbor pairs
    virtual void integrateOverAttachedNeighbors() = 0;      //!< Pure virtual function for integration over attached neighbor pairs
    virtual void integrateOverNotNeighbors() = 0;           //!< Pure virtual function for integration over non-neighbor pairs

    /*!
     * @brief Main procedure for integration of all possible pairs of neighbors
     * 
     * @param checkCorrectness Flag determining whether the integration error should be calculated after the integration procedure is finished
     */
    virtual void runAllPairs(bool checkCorrectness = false);

    /*!
     * @brief Procedure for integration of user-specified lists of tasks
     * 
     * @param userSimpleNeighborsTasks Vector of integration tasks for simple neighbor pairs
     * @param userAttachedNeighborsTasks Vector of integration tasks for attached neighbor pairs
     * @param userNotNeighborsTasks Vector of integration tasks for non-neighbor pairs
     * 
     * Vectors can be empty, in that case integration over neighbors of this type is not performed and no preparation is done.
     */
    void runPairs(const std::vector<int3> &userSimpleNeighborsTasks, const std::vector<int3> &userAttachedNeighborsTasks, const std::vector<int3> &userNotNeighborsTasks);

    /*!
     * @brief Save results of integration for a specified type of neighbor tasks into a file
     * 
     * @param neighborType Type of neighbor tasks for output
     * @param outputFormat Output file format (plain text or CSV)
     * @return true Results successfully saved to a file
     * @return false Error while saving results to a file
     */
    bool outputResultsToFile(neighbour_type_enum neighborType, output_format_enum outputFormat) const;

    /*!
     * @brief Get the vector of original integration tasks for a specific type of neighbor cells
     * 
     * @param neighborType Type of neighbors
     * @return const deviceVector<int3>* Pointer to device vector of tasks
     */
    const deviceVector<int3> *getTasks(neighbour_type_enum neighborType) const {
        switch(neighborType){
            case neighbour_type_enum::simple_neighbors:
                return &simpleNeighborsTasks;
            case neighbour_type_enum::attached_neighbors:
                return &attachedNeighborsTasks;
            case neighbour_type_enum::not_neighbors:
                return &notNeighborsTasks;
            default:
                return nullptr;
        }
    }

protected:
    /*!
     * @brief Compare integration results for 2 iterations using Runge rule
     * 
     * @param neighborType Type of neighbor pairs
     * @param allPairs Flag determining whether all pairs are checked or only those which did not converge after previous iteration
     * @return int Number of integrals which have not converged after the current iteration
     */
    int compareIntegrationResults(neighbour_type_enum neighborType, bool allPairs = false);

    /*!
     * @brief Device vector of original integration tasks for simple neighbor pairs
     * 
     * Each task is represented by an int3 value \f$(i,j,k)\f$, where \f$(i,j)\f$ corresponds to the indices of triangles
     * and \f$k\f$ is the index of pair in the whole vector (is used further in case of splitting of integral into several parts).
     */
    deviceVector<int3> simpleNeighborsTasks;

    /*!
     * @brief Device vector of original integration tasks for attached neighbor pairs
     * 
     * Each task is represented by an int3 value \f$(i,j,k)\f$, where \f$(i,j)\f$ corresponds to the indices of triangles
     * and \f$k\f$ is the index of pair in the whole vector (is used further in case of splitting of integral into several parts).
     */
    deviceVector<int3> attachedNeighborsTasks;

    /*!
     * @brief Device vector of original integration tasks for non-neighbor pairs
     * 
     * Each task is represented by an int3 value \f$(i,j,k)\f$, where \f$(i,j)\f$ corresponds to the indices of triangles
     * and \f$k\f$ is the index of pair in the whole vector (is used further in case of splitting of integral into several parts).
     */
    deviceVector<int3> notNeighborsTasks;

    /*!
     * @brief Device vector of integral values for original simple neighbor integration tasks
     * 
     * Includes both regular and singular parts. In case of mesh refinement this vector is gathered together from the integrals
     * over splitted triangles by using atomic summation. Integral value is stored using double4, where vector part corresponds
     * to the \f$\mathbf{\Psi}\f$ value and .w part is the \f$\Theta\f$ value.
     */
    deviceVector<double4> d_simpleNeighborsIntegrals;

    /*!
     * @brief Device vector of integral values for original attached neighbor integration tasks
     * 
     * Includes both regular and singular parts. In case of mesh refinement this vector is gathered together from the integrals
     * over splitted triangles by using atomic summation. Integral value is stored using double4, where vector part corresponds
     * to the \f$\mathbf{\Psi}\f$ value and .w part is the \f$\Theta\f$ value.
     */
    deviceVector<double4> d_attachedNeighborsIntegrals;
    
    /*!
     * @brief Device vector of integral values for original non-neighbor integration tasks
     * 
     * In case of mesh refinement this vector is gathered together from the integrals
     * over splitted triangles by using atomic summation. Integral value is stored using double4, where vector part corresponds
     * to the \f$\mathbf{\Psi}\f$ value and .w part is the \f$\Theta\f$ value.
     */
    deviceVector<double4> d_notNeighborsIntegrals;

    /*!
     * @brief Additional buffer with previous results of numerical integration of simple neighbor tasks
     * 
     * The vector is used for comparison of 2 refinement steps using Runge rule
     */
    deviceVector<double4> d_tempSimpleNeighborsIntegrals;

    /*!
     * @brief Additional buffer with previous results of numerical integration of attached neighbor tasks
     * 
     * The vector is used for comparison of 2 refinement steps using Runge rule
     */
    deviceVector<double4> d_tempAttachedNeighborsIntegrals;

    /*!
     * @brief Additional buffer with previous results of numerical integration of non-neighbor tasks
     * 
     * The vector is used for comparison of 2 refinement steps using Runge rule
     */
    deviceVector<double4> d_tempNotNeighborsIntegrals;

    /*!
     * @brief Indices of original tasks which have not yet converged and are left for further integration
     * for simple neighbors (for the next iteration)
     */
    deviceVector<int> simpleNeighborsTasksRest;

    /*!
     * @brief Indices of original tasks which have not yet converged and are left for further integration
     * for attached neighbors (for the next iteration)
     */
    deviceVector<int> attachedNeighborsTasksRest;

    /*!
     * @brief Indices of original tasks which have not yet converged and are left for further integration
     * for non-neighbors (for the next iteration)
     */
    deviceVector<int> notNeighborsTasksRest;

    /*!
     * @brief Buffer of indices of original tasks which have not yet converged and are left for further integration
     * for simple neighbors (for the current iteration)
     */
    deviceVector<int> tempSimpleNeighborsTasksRest;

    /*!
     * @brief Buffer of indices of original tasks which have not yet converged and are left for further integration
     * for attached neighbors (for the current iteration)
     */
    deviceVector<int> tempAttachedNeighborsTasksRest;
    
    /*!
     * @brief Buffer of indices of original tasks which have not yet converged and are left for further integration
     * for non-neighbors (for the current iteration)
     */
    deviceVector<int> tempNotNeighborsTasksRest;
    int *d_restTaskCount = nullptr;                                 //!< Counter of the tasks for which the integration procedure has not yet converged

    /*!
     * @brief Actual device vector of 3D values integration results for the simple neighbor tasks
     * 
     * These values are calculated from the double4 values (which contain separate values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$)
     * in the finalization procedure.
     */
    deviceVector<Point3> d_simpleNeighborsResults;

    /*!
     * @brief Actual device vector of 3D values integration results for the attached neighbor tasks
     * 
     * These values are calculated from the double4 values (which contain separate values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$)
     * in the finalization procedure.
     */
    deviceVector<Point3> d_attachedNeighborsResults;

    /*!
     * @brief Actual device vector of 3D values integration results for the non-neighbor tasks
     * 
     * These values are calculated from the double4 values (which contain separate values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$)
     * in the finalization procedure.
     */
    deviceVector<Point3> d_notNeighborsResults;

    /*!
     * @brief A Mesh3D object with the original surface mesh
     * 
     * Both vertex/cell data and neighbor pairs are used
     */
    const Mesh3D &mesh;

    NumericalIntegrator3D &numIntegrator;                           //!< A NumericalIntegrator3D object for numerical integration of the regular part of integrals

    GpuTimer timer;                                                 //!< Timer object for measurement and output of time of integration

private:
    deviceVector<double> simpleNeighborsErrors;                     //!< Vector of integration error calculated for \f$(i,j)\f$ and \f$(j,i)\f$ pairs for simple neighbor tasks
    deviceVector<double> attachedNeighborsErrors;                   //!< Vector of integration error calculated for \f$(i,j)\f$ and \f$(j,i)\f$ pairs for attached neighbor tasks
    deviceVector<double> notNeighborsErrors;                        //!< Vector of integration error calculated for \f$(i,j)\f$ and \f$(j,i)\f$ pairs for non-neighbor tasks
};

#endif // EVALUATOR3D_CUH
