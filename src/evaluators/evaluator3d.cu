#include "evaluator3d.cuh"

#include "../common/cuda_memory.cuh"

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
    simpleNeighborsTasks.free();
    d_simpleNeighborsResults.free();
    attachedNeighborsTasks.free();
    d_attachedNeighborsResults.free();
    notNeighborsTasks.free();
    d_notNeighborsResults.free();
}

void Evaluator3D::runAllPairs()
{
    int simpleNeighborsTasksNum = 2 * mesh.getSimpleNeighbors().size;
    int attachedNeighborsTasksNum = 2 * mesh.getAttachedNeighbors().size;
    int notNeighborsTasksNum = 2 * mesh.getNotNeighbors().size;

    simpleNeighborsTasks.allocate(simpleNeighborsTasksNum);
    d_simpleNeighborsResults.allocate(simpleNeighborsTasksNum);
    attachedNeighborsTasks.allocate(attachedNeighborsTasksNum);
    d_attachedNeighborsResults.allocate(attachedNeighborsTasksNum);
    notNeighborsTasks.allocate(notNeighborsTasksNum);
    d_notNeighborsResults.allocate(notNeighborsTasksNum);

    simpleNeighborsResults.resize(simpleNeighborsTasksNum);
    attachedNeighborsResults.resize(attachedNeighborsTasksNum);
    notNeighborsResults.resize(notNeighborsTasksNum);

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
