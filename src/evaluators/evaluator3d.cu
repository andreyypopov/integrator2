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

Evaluator3D::Evaluator3D(const Mesh3D &mesh_, const NumericalIntegrator3D *const numIntegrator_)
    : mesh(mesh_), numIntegrator(numIntegrator_)
{
}

Evaluator3D::~Evaluator3D()
{
    if(simpleNeighborsTasksNum){
        free_device(simpleNeighborsTasks);
        free_device(d_simpleNeighborsResults);
    }

    if(attachedNeighborsTasksNum){
        free_device(attachedNeighborsTasks);
        free_device(d_attachedNeighborsResults);
    }

    if(notNeighborsTasksNum){
        free_device(notNeighborsTasks);
        free_device(d_notNeighborsResults);
    }
}

void Evaluator3D::runAllPairs()
{
    simpleNeighborsTasksNum = 2 * mesh.getSimpleNeighborsNum();
    attachedNeighborsTasksNum = 2 * mesh.getAttachedNeighborsNum();
    notNeighborsTasksNum = 2 * mesh.getNotNeighborsNum();

    allocate_device(&simpleNeighborsTasks, simpleNeighborsTasksNum);
    allocate_device(&d_simpleNeighborsResults, simpleNeighborsTasksNum);
    allocate_device(&attachedNeighborsTasks, attachedNeighborsTasksNum);
    allocate_device(&d_attachedNeighborsResults, attachedNeighborsTasksNum);
    allocate_device(&notNeighborsTasks, notNeighborsTasksNum);
    allocate_device(&d_notNeighborsResults, notNeighborsTasksNum);

    simpleNeighborsResults.resize(simpleNeighborsTasksNum);
    attachedNeighborsResults.resize(attachedNeighborsTasksNum);
    notNeighborsResults.resize(notNeighborsTasksNum);

    copy_d2d(mesh.getSimpleNeighbors(), simpleNeighborsTasks, mesh.getSimpleNeighborsNum());
    copy_d2d(mesh.getAttachedNeighbors(), attachedNeighborsTasks, mesh.getAttachedNeighborsNum());
    copy_d2d(mesh.getNotNeighbors(), notNeighborsTasks, mesh.getNotNeighborsNum());

    unsigned int blocks;

    blocks = blocksForSize(mesh.getSimpleNeighborsNum());
    kAddReversedPairs<<<blocks, gpuThreads>>>(mesh.getSimpleNeighborsNum(), simpleNeighborsTasks);

    blocks = blocksForSize(mesh.getAttachedNeighborsNum());
    kAddReversedPairs<<<blocks, gpuThreads>>>(mesh.getAttachedNeighborsNum(), attachedNeighborsTasks);

    blocks = blocksForSize(mesh.getNotNeighborsNum());
    kAddReversedPairs<<<blocks, gpuThreads>>>(mesh.getNotNeighborsNum(), notNeighborsTasks);
    
    integrateOverSimpleNeighbors();
    integrateOverAttachedNeighbors();
    integrateOverNotNeighbors();
}
