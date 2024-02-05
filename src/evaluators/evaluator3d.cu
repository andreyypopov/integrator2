#include "evaluator3d.cuh"

#include "../common/cuda_memory.cuh"

#include <fstream>

__global__ void kAddReversedPairs(int n, int2 *pairs)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        int2 oldPair = pairs[idx];
        int2 newPair;
        newPair.x = oldPair.y;
        newPair.y = oldPair.x;

        pairs[n + idx] = newPair;
    }
}

Evaluator3D::Evaluator3D(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_)
    : mesh(mesh_), numIntegrator(numIntegrator_)
{
    errorControlType = error_control_type_enum::automatic_error_control;
}

Evaluator3D::~Evaluator3D()
{

}

void Evaluator3D::runAllPairs()
{
    int simpleNeighborsTasksNum = 2 * mesh.getSimpleNeighbors().size;
    int attachedNeighborsTasksNum = 2 * mesh.getAttachedNeighbors().size;
    int notNeighborsTasksNum = 2 * mesh.getNotNeighbors().size;

    simpleNeighborsTasks.allocate(simpleNeighborsTasksNum);
    d_simpleNeighborsResults.allocate(simpleNeighborsTasksNum);
    d_simpleNeighborsIntegrals.allocate(simpleNeighborsTasksNum);
    attachedNeighborsTasks.allocate(attachedNeighborsTasksNum);
    d_attachedNeighborsResults.allocate(attachedNeighborsTasksNum);
    d_attachedNeighborsIntegrals.allocate(attachedNeighborsTasksNum);
    notNeighborsTasks.allocate(notNeighborsTasksNum);
    d_notNeighborsResults.allocate(notNeighborsTasksNum);
    d_notNeighborsIntegrals.allocate(notNeighborsTasksNum);

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
    
    if(errorControlType == error_control_type_enum::fixed_refinement_level)
        numIntegrator.prepareTasksAndRefineWholeMesh(simpleNeighborsTasks, attachedNeighborsTasks, notNeighborsTasks, meshRefinementLevel);

    integrateOverSimpleNeighbors();
    integrateOverAttachedNeighbors();
    integrateOverNotNeighbors();
}

void Evaluator3D::setFixedRefinementLevel(int refinementLevel)
{
    errorControlType = error_control_type_enum::fixed_refinement_level;
    meshRefinementLevel = refinementLevel;
}

bool Evaluator3D::outputResultsToFile(neighbour_type_enum neighborType) const
{
    int2 *tasks;
    int tasksSize;
    Point3 *deviceResults;
    std::string filename;

    switch (neighborType)
    {
    case neighbour_type_enum::simple_neighbors:
        tasks = simpleNeighborsTasks.data;
        tasksSize = simpleNeighborsTasks.size;
        deviceResults = d_simpleNeighborsResults.data;
        filename = "SimpleNeighbors.dat";
        break;
    case neighbour_type_enum::attached_neighbors:
        tasks = attachedNeighborsTasks.data;
        tasksSize = attachedNeighborsTasks.size;
        deviceResults = d_attachedNeighborsResults.data;
        filename = "AttachedNeighbors.dat";
        break;
    case neighbour_type_enum::not_neighbors:
        tasks = notNeighborsTasks.data;
        tasksSize = notNeighborsTasks.size;
        deviceResults = d_notNeighborsResults.data;
        filename = "NotNeighbors.dat";
        break;
    }

    std::vector<Point3> hostResults(tasksSize);
    std::vector<int2> hostTasks(tasksSize);

    copy_d2h(deviceResults, hostResults.data(), tasksSize);
    copy_d2h(tasks, hostTasks.data(), tasksSize);

    std::ofstream resultsFile(filename.c_str());

    if(resultsFile.is_open()){
        for(int i = 0; i < tasksSize; ++i)
            resultsFile << "(" << hostTasks[i].x << ", " << hostTasks[i].y << "): ["
                    << hostResults[i].x << ", " << hostResults[i].y << ", " << hostResults[i].z << "]" << std::endl;

        resultsFile.close();

        printf("%d results saved to file %s\n", tasksSize, filename.c_str());

        return true;
    } else {
        printf("Error while opening the file\n");
        return false;
    }
}
