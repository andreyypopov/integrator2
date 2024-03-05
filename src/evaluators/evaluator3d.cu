#include "evaluator3d.cuh"

#include "../common/cuda_memory.cuh"

#include <fstream>

__constant__ double c_pow2p;

__global__ void kAddReversedPairs(int n, int3 *pairs)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int3 oldPair = pairs[idx];
        int3 newPair = { oldPair.y, oldPair.x, (int)(n + idx) };

        pairs[n + idx] = newPair;
    }
}

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

    integrateOverSimpleNeighbors();
    integrateOverAttachedNeighbors();
    integrateOverNotNeighbors();

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

    if(!userSimpleNeighborsTasks.empty())
        integrateOverSimpleNeighbors();

    if(!userAttachedNeighborsTasks.empty())
        integrateOverAttachedNeighbors();

    if(!userNotNeighborsTasks.empty())
        integrateOverNotNeighbors();    
}

int Evaluator3D::compareIntegrationResults(neighbour_type_enum neighborType, bool allPairs)
{
    zero_value_device(d_restTaskCount, 1);
    deviceVector<double4> *integrals, *tempIntegrals;
    deviceVector<int> *restTasks, *tempRestTasks;
    unsigned char *tasksConverged = numIntegrator.getIntegralsConverged(neighborType).data;

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
    restTasks->size = notConvergedTaskCount;
    
    printf("Out of %d tasks: %d converged, %d did not converge\n", taskCount, taskCount - notConvergedTaskCount, notConvergedTaskCount);
    
    return notConvergedTaskCount;
}

bool Evaluator3D::outputResultsToFile(neighbour_type_enum neighborType) const
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
        filename = "SimpleNeighbors.dat";
        break;
    case neighbour_type_enum::attached_neighbors:
        tasks = attachedNeighborsTasks.data;
        tasksSize = attachedNeighborsTasks.size;
        deviceResults = d_attachedNeighborsResults.data;
        if(attachedNeighborsErrors.data)
            errors = attachedNeighborsErrors.data;
        filename = "AttachedNeighbors.dat";
        break;
    case neighbour_type_enum::not_neighbors:
        tasks = notNeighborsTasks.data;
        tasksSize = notNeighborsTasks.size;
        deviceResults = d_notNeighborsResults.data;
        if(notNeighborsErrors.data)
            errors = notNeighborsErrors.data;
        filename = "NotNeighbors.dat";
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
        for(int i = 0; i < tasksSize; ++i){
            resultsFile << "(" << hostTasks[i].x << ", " << hostTasks[i].y << "): ["
                    << hostResults[i].x << ", " << hostResults[i].y << ", " << hostResults[i].z << "]";

            if(errors)
                resultsFile << ", error = " << hostErrors[i];

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
