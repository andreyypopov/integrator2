/*!
 * @file evaluator3d.cu
 * @brief Implementation of the Evaluator3D class, including the overall procedure of integration, comparison of results of 2 iterations
 * (using Runge rule) and output of results to plain text or CSV.
 */
#include "evaluator3d.cuh"

#include "../common/cuda_memory.cuh"

#include <fstream>

__constant__ double c_pow2p;        //!< Pre-computed value of \f$2^p\f$ for use in the Runge rule

/*!
 * @brief Kernel function for complementing the initial list of integration tasks with reversed pairs
 * 
 * @param n Number of integration tasks
 * @param pairs List of integration task (pair of triangle indices + index of the pair)
 *
 * First \f$n\f$ values are processed and for each a new task is added with triangle indices reversed.
 */
__global__ void kAddReversedPairs(int n, int3 *pairs)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int3 oldPair = pairs[idx];
        int3 newPair = { oldPair.y, oldPair.x, (int)(n + idx) };

        pairs[n + idx] = newPair;
    }
}

/*!
 * @brief Kernel function for calculation of integration error by comparing results for \f$(i,j)\f$ and \f$(j,i)\f$ pairs
 * 
 * @param n Number of original neighbor pairs (only \f$(i,j),\,i<j\f$)
 * @param errors Vector of integration error values
 * @param results Integration results (vector of \f$2n\f$ length)
 * 
 * Error is calculate using the formula
 * \f[
 *      \delta = \frac{\|\mathbf J_{3D}(K_i,\,K_j) + \mathbf J_{3D}(K_j,\,K_i)\|}{\max\{\|\mathbf J_{3D}(K_i,\,K_j)\|, \|\mathbf J_{3D}(K_j,\,K_i)\|\}}.
 * \f]
 */
__global__ void kCalculateIntegrationError(int n, double *errors, const Point3 *results)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const Point3 resultIJ = results[idx];
        const Point3 resultJI = results[n + idx];

        const double delta = norm1(resultIJ + resultJI) / max(norm1(resultIJ), norm1(resultJI));

        errors[idx] = delta;
        errors[n + idx] = delta;
    }
}

/*!
 * @brief Kernel function which checks the Runge rule and determines whether the integration process has converged or further computation for this task is needed
 * 
 * @param n Number of integration tasks to be checked
 * @param integrals Vector of integration results on the current iteration
 * @param tempIntegrals Vector of integration results on the previous iteration
 * @param d_restTaskCount Counter of the integration tasks which have not converged yet
 * @param restTasks Indices of tasks which have not yet converged
 * @param tempRestTasks Indices of tasks to be integrated (a list filled on the previous iteration or nullptr, then all tasks are checked)
 * @param taskConverged Vector of flags which indicate whether integration task has converged or not
 * 
 * Both current integral values and values from the previous iteration have the same length which is equal to the number of original tasks.
 * Runge rule criterion is calculated component-wise for all 4 components of integral value, then norm is calculated and compared with tolerance.
 * It is crucial to mark converged tasks with a flag so as not to calculate them again in order to spare computation time.
 * There may be a situation when a cell is further refined and some of the integrals where it is \f$i\f$-th cell have converged
 * (and should not be further calculated) and some have not and require splitting.
 */
__global__ void kCompareIntegrationResults(int n, const double4 *integrals, const double4 *tempIntegrals, int *d_restTaskCount, int *restTasks, const int *tempRestTasks, unsigned char *taskConverged)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        if(tempRestTasks)
            idx = tempRestTasks[idx];

        const double4 integralValue = integrals[idx];
        const double4 refinedIntegralValue = tempIntegrals[idx];

        const double4 numerator = integralValue - refinedIntegralValue;
        const double4 denominator = c_pow2p * refinedIntegralValue - integralValue;
        const double4 div = divide(numerator, denominator);

        const double criterion = norm1(div);

        if(criterion > CONSTANTS::EPS_INTEGRATION){
            int pos = atomicAdd(d_restTaskCount, 1);
            restTasks[pos] = idx;
            taskConverged[idx] = false;
        } else
            taskConverged[idx] = true;
    }
}

Evaluator3D::Evaluator3D(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_)
    : mesh(mesh_), numIntegrator(numIntegrator_)
{
    const double pow2p = 1 << numIntegrator.getQuadratureFormulaOrder();
    copy_h2const(&pow2p, &c_pow2p, 1);
}

Evaluator3D::~Evaluator3D()
{
    free_device(d_restTaskCount);
}

/*!
 * -# Necessary memory allocations are performed (including additional vectors if adaptive error control is used).
 * -# Neighbor pairs are directly copied to vectors of tasks and filled with reversed pairs using a specific kernel function.
 * -# Necessary preparation for numerical integration is performed (memory allocations, mesh refinement, etc.).
 * -# Integration procedures are called for different types (should be implemented in the child classes).
 * -# Integration error can optionally be calculated by comparing integrals for \f$(i,j)\f$ and \f$(j,i)\f$ pairs.
 */
void Evaluator3D::runAllPairs(bool checkCorrectness)
{
    int simpleNeighborsTasksNum = 2 * mesh.getSimpleNeighbors().size;
    int attachedNeighborsTasksNum = 2 * mesh.getAttachedNeighbors().size;
    int notNeighborsTasksNum = 2 * mesh.getNotNeighbors().size;

    simpleNeighborsTasks.allocate(simpleNeighborsTasksNum);
    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
        simpleNeighborsTasksRest.allocate(simpleNeighborsTasksNum);
        tempSimpleNeighborsTasksRest.allocate(simpleNeighborsTasksNum);
        d_tempSimpleNeighborsIntegrals.allocate(simpleNeighborsTasksNum);
    }
    d_simpleNeighborsResults.allocate(simpleNeighborsTasksNum);
    d_simpleNeighborsIntegrals.allocate(simpleNeighborsTasksNum);
    
    attachedNeighborsTasks.allocate(attachedNeighborsTasksNum);
    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
        attachedNeighborsTasksRest.allocate(attachedNeighborsTasksNum);
        tempAttachedNeighborsTasksRest.allocate(attachedNeighborsTasksNum);
        d_tempAttachedNeighborsIntegrals.allocate(attachedNeighborsTasksNum);
    }
    d_attachedNeighborsResults.allocate(attachedNeighborsTasksNum);
    d_attachedNeighborsIntegrals.allocate(attachedNeighborsTasksNum);
    
    notNeighborsTasks.allocate(notNeighborsTasksNum);
    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
        notNeighborsTasksRest.allocate(notNeighborsTasksNum);
        tempNotNeighborsTasksRest.allocate(notNeighborsTasksNum);
        d_tempNotNeighborsIntegrals.allocate(notNeighborsTasksNum);
    }
    d_notNeighborsResults.allocate(notNeighborsTasksNum);
    d_notNeighborsIntegrals.allocate(notNeighborsTasksNum);
    
    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control)
        allocate_device(&d_restTaskCount, 1);

    copy_d2d(mesh.getSimpleNeighbors().data, simpleNeighborsTasks.data, mesh.getSimpleNeighbors().size);
    copy_d2d(mesh.getAttachedNeighbors().data, attachedNeighborsTasks.data, mesh.getAttachedNeighbors().size);
    copy_d2d(mesh.getNotNeighbors().data, notNeighborsTasks.data, mesh.getNotNeighbors().size);

    unsigned int blocks;

    blocks = blocksForSize(mesh.getSimpleNeighbors().size);
    kAddReversedPairs<<<blocks, gpuThreads>>>(mesh.getSimpleNeighbors().size, simpleNeighborsTasks.data);

    blocks = blocksForSize(mesh.getAttachedNeighbors().size);
    kAddReversedPairs<<<blocks, gpuThreads>>>(mesh.getAttachedNeighbors().size, attachedNeighborsTasks.data);

    blocks = blocksForSize(mesh.getNotNeighbors().size);
    kAddReversedPairs<<<blocks, gpuThreads>>>(mesh.getNotNeighbors().size, notNeighborsTasks.data);
    
    numIntegrator.prepareTasksAndMesh(simpleNeighborsTasks, attachedNeighborsTasks, notNeighborsTasks);

    timer.start();
    integrateOverSimpleNeighbors();
    timer.stop("Simple neighbors integration");
    requestFreeDeviceMemoryAmount();

    timer.start();
    integrateOverAttachedNeighbors();
    timer.stop("Attached neighbors integration");
    requestFreeDeviceMemoryAmount();

    timer.start();    
    integrateOverNotNeighbors();
    timer.stop("Non-neighbors integration");
    requestFreeDeviceMemoryAmount();

    if(checkCorrectness){
        simpleNeighborsErrors.allocate(simpleNeighborsTasksNum);
        attachedNeighborsErrors.allocate(attachedNeighborsTasksNum);
        notNeighborsErrors.allocate(notNeighborsTasksNum);

        blocks = blocksForSize(mesh.getSimpleNeighbors().size);
        kCalculateIntegrationError<<<blocks, gpuThreads>>>(mesh.getSimpleNeighbors().size, simpleNeighborsErrors.data, d_simpleNeighborsResults.data);

        blocks = blocksForSize(mesh.getAttachedNeighbors().size);
        kCalculateIntegrationError<<<blocks, gpuThreads>>>(mesh.getAttachedNeighbors().size, attachedNeighborsErrors.data, d_attachedNeighborsResults.data);

        blocks = blocksForSize(mesh.getNotNeighbors().size);
        kCalculateIntegrationError<<<blocks, gpuThreads>>>(mesh.getNotNeighbors().size, notNeighborsErrors.data, d_notNeighborsResults.data);

        checkCudaErrors(cudaDeviceSynchronize());
    }
}

/*!
 * -# Memory allocation is performed for types of neighbors for which the lists of tasks are non-empty. Tasks are copied from host to device memory.
 * -# Necessary preparation for numerical integration is performed (memory allocations, mesh refinement, etc.).
 * -# Integration procedures are called for types with non-empty lists (should be implemented in the child classes).
 * 
 * Integration error is not calculated as a list may not contain the \f$(i,j)\f$ and \f$(j,i)\f$ pairs at the same time.
 */
void Evaluator3D::runPairs(const std::vector<int3> &userSimpleNeighborsTasks, const std::vector<int3> &userAttachedNeighborsTasks, const std::vector<int3> &userNotNeighborsTasks)
{
    if(userSimpleNeighborsTasks.empty() && userAttachedNeighborsTasks.empty() && userNotNeighborsTasks.empty())
        return;

    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control)
        allocate_device(&d_restTaskCount, 1);

    if(!userSimpleNeighborsTasks.empty()){
        const int simpleNeighborsTasksNum = userSimpleNeighborsTasks.size();
        simpleNeighborsTasks.allocate(simpleNeighborsTasksNum);

        if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
            simpleNeighborsTasksRest.allocate(simpleNeighborsTasksNum);
            tempSimpleNeighborsTasksRest.allocate(simpleNeighborsTasksNum);
            d_tempSimpleNeighborsIntegrals.allocate(simpleNeighborsTasksNum);
        }
        d_simpleNeighborsResults.allocate(simpleNeighborsTasksNum);
        d_simpleNeighborsIntegrals.allocate(simpleNeighborsTasksNum);
        
        copy_h2d(userSimpleNeighborsTasks.data(), simpleNeighborsTasks.data, simpleNeighborsTasksNum);
    }
    
    if(!userAttachedNeighborsTasks.empty()){
        const int attachedNeighborsTasksNum = userAttachedNeighborsTasks.size();
        attachedNeighborsTasks.allocate(attachedNeighborsTasksNum);

        if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
            attachedNeighborsTasksRest.allocate(attachedNeighborsTasksNum);
            tempAttachedNeighborsTasksRest.allocate(attachedNeighborsTasksNum);
            d_tempAttachedNeighborsIntegrals.allocate(attachedNeighborsTasksNum);
        }
        d_attachedNeighborsResults.allocate(attachedNeighborsTasksNum);
        d_attachedNeighborsIntegrals.allocate(attachedNeighborsTasksNum);
        
        copy_h2d(userAttachedNeighborsTasks.data(), attachedNeighborsTasks.data, attachedNeighborsTasksNum);
    }
    
    if(!userNotNeighborsTasks.empty()){
        const int notNeighborsTasksNum = userNotNeighborsTasks.size();
        notNeighborsTasks.allocate(notNeighborsTasksNum);
        
        if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
            notNeighborsTasksRest.allocate(notNeighborsTasksNum);
            tempNotNeighborsTasksRest.allocate(notNeighborsTasksNum);
            d_tempNotNeighborsIntegrals.allocate(notNeighborsTasksNum);
        }
        d_notNeighborsResults.allocate(notNeighborsTasksNum);
        d_notNeighborsIntegrals.allocate(notNeighborsTasksNum);
        
        copy_h2d(userNotNeighborsTasks.data(), notNeighborsTasks.data, notNeighborsTasksNum);
    }

    numIntegrator.prepareTasksAndMesh(simpleNeighborsTasks, attachedNeighborsTasks, notNeighborsTasks);

    if(!userSimpleNeighborsTasks.empty()){
        timer.start();
        integrateOverSimpleNeighbors();
        timer.stop("Simple neighbors integration");
        requestFreeDeviceMemoryAmount();
    }

    if(!userAttachedNeighborsTasks.empty()){
        timer.start();
        integrateOverAttachedNeighbors();
        timer.stop("Attached neighbors integration");
        requestFreeDeviceMemoryAmount();
    }

    if(!userNotNeighborsTasks.empty()){
        timer.start();
        integrateOverNotNeighbors();
        timer.stop("Non-neighbors integration");
        requestFreeDeviceMemoryAmount();
    }    
}

/*!
 * A kernel function is called for 2 vectors of integral values (double4), which are compared using the Runge rule:
 * \f$\varepsilon = \left\|\frac{I_h - I_{h/2}}{2^p I_{h/2} - I_h}\right\|\f$, where \f$I_h\f$ and \f$I_{h/2}\f$ are values (each separate component)
 * of the integrals for previous and current iterations of refinement, respectively. After 2 iterations (first time comparison is performed)
 * all the integrals are compared, then the list is formed of indices of integrals which have not converged. The function
 * prints the message and returns the number of tasks which have converged after current iteration.
 */
int Evaluator3D::compareIntegrationResults(neighbour_type_enum neighborType, bool allPairs)
{
    zero_value_device(d_restTaskCount, 1);
    deviceVector<double4> *integrals, *tempIntegrals;
    deviceVector<int> *restTasks, *tempRestTasks;
    unsigned char *tasksConverged = numIntegrator.getIntegralsConverged(neighborType)->data;

    switch (neighborType)
    {
    case neighbour_type_enum::simple_neighbors:
        integrals = &d_simpleNeighborsIntegrals;
        tempIntegrals = &d_tempSimpleNeighborsIntegrals;
        restTasks = &simpleNeighborsTasksRest;
        tempRestTasks = &tempSimpleNeighborsTasksRest;
        break;
    case neighbour_type_enum::attached_neighbors:
        integrals = &d_attachedNeighborsIntegrals;
        tempIntegrals = &d_tempAttachedNeighborsIntegrals;
        restTasks = &attachedNeighborsTasksRest;
        tempRestTasks = &tempAttachedNeighborsTasksRest;
        break;
    case neighbour_type_enum::not_neighbors:
        integrals = &d_notNeighborsIntegrals;
        tempIntegrals = &d_tempNotNeighborsIntegrals;
        restTasks = &notNeighborsTasksRest;
        tempRestTasks = &tempNotNeighborsTasksRest;
        break;
    }

    restTasks->swap(*tempRestTasks);

    const int taskCount = allPairs ? integrals->size : tempRestTasks->size;
    unsigned int blocks = blocksForSize(taskCount);
    kCompareIntegrationResults<<<blocks, gpuThreads>>>(taskCount, integrals->data, tempIntegrals->data, d_restTaskCount, restTasks->data, allPairs ? nullptr : tempRestTasks->data, tasksConverged);

    checkCudaErrors(cudaDeviceSynchronize());
    int notConvergedTaskCount;
    
    copy_d2h(d_restTaskCount, &notConvergedTaskCount, 1);
    restTasks->resize(notConvergedTaskCount);
    
    printf("Out of %d tasks: %d converged, %d did not converge\n", taskCount, taskCount - notConvergedTaskCount, notConvergedTaskCount);
    
    return notConvergedTaskCount;
}

/*!
 * The following data is exported for each integral:
 * - task - pair \f$(i,j)\f$;
 * - integral value - 3D vector;
 * - (optional) integration error by comparing with the \f$(j,i)\f$ pair, if previously computed.
 * 
 * These 3 (or only 2, if error is not exported) vectors are copied to host from device memory.
 * 
 * In CSV format these values are output sequentually with separators placed between them.
 * In plain text format the values are output in the following format: "(i,j): [Ix, Iy, Iz], error = value".
 */
bool Evaluator3D::outputResultsToFile(neighbour_type_enum neighborType, output_format_enum outputFormat) const
{
    int3 *tasks;
    int tasksSize;
    Point3 *deviceResults;
    double *errors = nullptr;
    std::string filename;

    switch (neighborType)
    {
    case neighbour_type_enum::simple_neighbors:
        tasks = simpleNeighborsTasks.data;
        tasksSize = simpleNeighborsTasks.size;
        deviceResults = d_simpleNeighborsResults.data;
        if(simpleNeighborsErrors.data)
            errors = simpleNeighborsErrors.data;
        filename = "SimpleNeighbors";
        break;
    case neighbour_type_enum::attached_neighbors:
        tasks = attachedNeighborsTasks.data;
        tasksSize = attachedNeighborsTasks.size;
        deviceResults = d_attachedNeighborsResults.data;
        if(attachedNeighborsErrors.data)
            errors = attachedNeighborsErrors.data;
        filename = "AttachedNeighbors";
        break;
    case neighbour_type_enum::not_neighbors:
        tasks = notNeighborsTasks.data;
        tasksSize = notNeighborsTasks.size;
        deviceResults = d_notNeighborsResults.data;
        if(notNeighborsErrors.data)
            errors = notNeighborsErrors.data;
        filename = "NotNeighbors";
        break;
    }

    switch (outputFormat)
    {
    case output_format_enum::plainText:
        filename += ".dat";
        break;    
    case output_format_enum::csv:
        filename += ".csv";
        break;
    }

    if(!tasksSize)
        return false;

    std::vector<Point3> hostResults(tasksSize);
    std::vector<int3> hostTasks(tasksSize);
    std::vector<double> hostErrors;

    copy_d2h(deviceResults, hostResults.data(), tasksSize);
    copy_d2h(tasks, hostTasks.data(), tasksSize);

    if(errors){
        hostErrors.resize(tasksSize);
        copy_d2h(errors, hostErrors.data(), tasksSize);
    }

    std::ofstream resultsFile(filename.c_str());

    if(resultsFile.is_open()){
        //header for csv
        if(outputFormat == output_format_enum::csv){
            resultsFile << "\"TaskI\";\"TaskJ\";\"IntegralX\";\"IntegralY\";\"IntegralZ\"";
            if(errors)
                resultsFile << ";\"Error\"";
            resultsFile << std::endl;
        }

        for(int i = 0; i < tasksSize; ++i){
            switch (outputFormat)
            {
            case output_format_enum::plainText:
                resultsFile << "(" << hostTasks[i].x << ", " << hostTasks[i].y << "): ["
                    << hostResults[i].x << ", " << hostResults[i].y << ", " << hostResults[i].z << "]";

                if(errors)
                    resultsFile << ", error = " << hostErrors[i];
                break;
            case output_format_enum::csv:
                resultsFile << hostTasks[i].x << ";" << hostTasks[i].y << ";"
                    << hostResults[i].x << ";" << hostResults[i].y << ";" << hostResults[i].z;

                if(errors)
                    resultsFile << ";" << hostErrors[i];
                break;
            }

            resultsFile << std::endl;
        }

        resultsFile.close();

        printf("%d results saved to file %s\n", tasksSize, filename.c_str());

        return true;
    } else {
        printf("Error while opening the file\n");
        return false;
    }
}
