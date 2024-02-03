#include "NumericalIntegrator3d.cuh"

#include "common/cuda_memory.cuh"

__constant__ Point3 c_GaussPointsCoordinates[CONSTANTS::MAX_GAUSS_POINTS];
__constant__ double c_GaussPointsWeights[CONSTANTS::MAX_GAUSS_POINTS];

__global__ void kSplitCell(int n, Point3 *refinedVertices, int3 *refinedCells, double *refinedCellMeasures, int *originalCells, int2 *refinedVerticesCellsNum,
        const Point3 *tempVertices, const int3 *tempCells, const double *tempCellMeasures, const int *tempOriginalCells)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        //data of the triangle to be split
        const int3 triangle = tempCells[idx];
        const Point3 triA = tempVertices[triangle.x];
        const Point3 triB = tempVertices[triangle.y];
        const Point3 triC = tempVertices[triangle.z];
        const double measure = tempCellMeasures[idx];
        const int originalCellIndex = tempOriginalCells[idx];

        //add 3 new vertices on the triangle egdes
        const int vertexIndex = atomicAdd((int*)refinedVerticesCellsNum, 3);
        refinedVertices[vertexIndex] = 0.5 * (triB + triC);
        refinedVertices[vertexIndex + 1] = 0.5 * (triC + triA);
        refinedVertices[vertexIndex + 2] = 0.5 * (triA + triB);

        //create 4 new triangles
        const int cellIndex = atomicAdd((int*)refinedVerticesCellsNum + 1, 4);
        refinedCells[cellIndex] = { vertexIndex + 2, triangle.y, vertexIndex };
        refinedCells[cellIndex + 1] = { vertexIndex, triangle.z, vertexIndex + 1 };
        refinedCells[cellIndex + 2] = { vertexIndex + 1, triangle.x, vertexIndex + 2 };
        refinedCells[cellIndex + 3] = { vertexIndex, vertexIndex + 1, vertexIndex + 2 };

        //determine measure of each new cell
        const double newMeasure = 0.25 * measure;
        refinedCellMeasures[cellIndex] = newMeasure;
        refinedCellMeasures[cellIndex + 1] = newMeasure;
        refinedCellMeasures[cellIndex + 2] = newMeasure;
        refinedCellMeasures[cellIndex + 3] = newMeasure;

        //update correspondence to the original cell index
        originalCells[cellIndex] = originalCellIndex;
        originalCells[cellIndex + 1] = originalCellIndex;
        originalCells[cellIndex + 2] = originalCellIndex;
        originalCells[cellIndex + 3] = originalCellIndex;
    }
}

__global__ void kCountOrCreateTasks(int tasksNum, int refinedCellsNum, int *counter, const int2 *tasks, const int *originalCells, int3 *refinedTasks = nullptr)
{
    int taskId = blockIdx.x * blockDim.x + threadIdx.x;
    int refinedCellId = blockIdx.y * blockDim.y + threadIdx.y;

    if(taskId < tasksNum && refinedCellId < refinedCellsNum){
        const int2 oldTask = tasks[taskId];
        const int originalCell = originalCells[refinedCellId];

        if(oldTask.x == originalCell){
            int pos = atomicAdd(counter, 1);
            if(refinedTasks)
                refinedTasks[pos] = { refinedCellId, oldTask.y, taskId };
        }
    }
}

__global__ void kCopyTasks(int n, int3 *refinedTasks, const int2 *tasks)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int2 oldTask = tasks[idx];
        refinedTasks[idx] = { oldTask.x, oldTask.y, (int)idx };
    }
}

__global__ void kSumIntegrationResults(int n, double4 *results, const double4 *refinedResults, const int3 *refinedTasks)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = refinedTasks[idx];
        const double4 refinedResult = refinedResults[idx];
        double *originalResult = (double*)(results + task.z);

        atomicAdd(originalResult + 0, refinedResult.x);
        atomicAdd(originalResult + 1, refinedResult.y);
        atomicAdd(originalResult + 2, refinedResult.z);
        atomicAdd(originalResult + 3, refinedResult.w);
    }
}

__global__ void kInitializeOriginalCellIndices(int n, int *originalCells)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        originalCells[idx] = idx;
}

NumericalIntegrator3D::NumericalIntegrator3D(const Mesh3D &mesh_, const QuadratureFormula3D &qf_)
    : GaussPointsNum(qf_.weights.size()), mesh(mesh_), qf(qf_)
{
    std::vector<Point3> lCoordinates(GaussPointsNum);

    for(int i = 0; i < GaussPointsNum; ++i){
        lCoordinates[i].x = qf_.coordinates[i].x;
        lCoordinates[i].y = qf_.coordinates[i].y;
        lCoordinates[i].z = 1.0 - qf_.coordinates[i].x - qf_.coordinates[i].y;
    }

    copy_h2const(lCoordinates.data(), c_GaussPointsCoordinates, GaussPointsNum);
    copy_h2const(qf.weights.data(), c_GaussPointsWeights, GaussPointsNum);
}

NumericalIntegrator3D::~NumericalIntegrator3D()
{

}

void NumericalIntegrator3D::prepareTasksAndRefineWholeMesh(const deviceVector<int2> &simpleNeighborsTasks, const deviceVector<int2> &attachedNeighborsTasks, const deviceVector<int2> &notNeighborsTasks, int refineLevel)
{
    int2 verticesCellsNum = { mesh.getVertices().size, mesh.getCells().size };

    if(!refineLevel){
        //copy mesh vertices, cells and measures as is
        refinedVertices.allocate(verticesCellsNum.x);
        copy_d2d(mesh.getVertices().data, refinedVertices.data, verticesCellsNum.x);

        refinedCells.allocate(verticesCellsNum.y);
        copy_d2d(mesh.getCells().data, refinedCells.data, verticesCellsNum.y);
        
        refinedCellMeasures.allocate(verticesCellsNum.y);
        copy_d2d(mesh.getCellMeasures().data, refinedCellMeasures.data, verticesCellsNum.y);
        
        //copy tasks and prepare vectors for results
        if(simpleNeighborsTasks.size){
            refinedSimpleNeighborsTasks.allocate(simpleNeighborsTasks.size);
            
            int blocks = blocksForSize(simpleNeighborsTasks.size);
            kCopyTasks<<<blocks, gpuThreads>>>(simpleNeighborsTasks.size, refinedSimpleNeighborsTasks.data, simpleNeighborsTasks.data);
            d_simpleNeighborsResults.allocate(simpleNeighborsTasks.size);
        }

        if(attachedNeighborsTasks.size){
            refinedAttachedNeighborsTasks.allocate(attachedNeighborsTasks.size);

            int blocks = blocksForSize(attachedNeighborsTasks.size);
            kCopyTasks<<<blocks, gpuThreads>>>(attachedNeighborsTasks.size, refinedAttachedNeighborsTasks.data, attachedNeighborsTasks.data);

            d_simpleNeighborsResults.allocate(attachedNeighborsTasks.size);
        }

        if(notNeighborsTasks.size){
            refinedNotNeighborsTasks.allocate(notNeighborsTasks.size);

            int blocks = blocksForSize(notNeighborsTasks.size);
            kCopyTasks<<<blocks, gpuThreads>>>(notNeighborsTasks.size, refinedNotNeighborsTasks.data, notNeighborsTasks.data);

            d_simpleNeighborsResults.allocate(notNeighborsTasks.size);
        }

        return;
    }

    int refinedVerticesNum = verticesCellsNum.x + verticesCellsNum.y * ((1 << (2 * refineLevel)) - 1);
    refinedVertices.allocate(refinedVerticesNum);
    tempVertices.allocate(refinedVerticesNum);

    int refinedCellsNum = (1 << (2 * refineLevel)) * verticesCellsNum.y;
    refinedCells.allocate(refinedCellsNum);
    tempCells.allocate(refinedCellsNum);
    refinedCellMeasures.allocate(refinedCellsNum);
    tempCellMeasures.allocate(refinedCellsNum);

    //vectors for indices of original triangles (with respect to the refined triangles)
    deviceVector<int> originalCells, tempOriginalCells;
    originalCells.allocate(refinedCellsNum);
    tempOriginalCells.allocate(refinedCellsNum);

    //initialize vectors using original data (will then be moved to the temporary buffers)
    copy_d2d(mesh.getVertices().data, refinedVertices.data, verticesCellsNum.x);
    copy_d2d(mesh.getCells().data, refinedCells.data, verticesCellsNum.y);
    copy_d2d(mesh.getCellMeasures().data, refinedCellMeasures.data, verticesCellsNum.y);

    //initialize number of indices of original cells with 0,1,2,...,ncells-1
    unsigned int blocks = blocksForSize(verticesCellsNum.y);
    kInitializeOriginalCellIndices<<<blocks, gpuThreads>>>(verticesCellsNum.y, originalCells.data);

    int2 *refinedVerticesCellsNum;
    allocate_device(&refinedVerticesCellsNum, 1);
    copy_h2d(&verticesCellsNum, refinedVerticesCellsNum, 1);

    for(int i = 0; i < refineLevel; ++i){
        std::swap(refinedVertices.data, tempVertices.data);
        std::swap(refinedCells.data, tempCells.data);
        std::swap(refinedCellMeasures.data, tempCellMeasures.data);
        std::swap(originalCells.data, tempOriginalCells.data);

        //vertex list from the previous iteration forms the first part of the new vertex list
        //leave the vertex count equal to the size of the previous vertex list and set the cell count to zero
        copy_d2d(tempVertices.data, refinedVertices.data, verticesCellsNum.x);
        zero_value_device((int*)refinedVerticesCellsNum + 1, 1);

        blocks = blocksForSize(verticesCellsNum.y);
        kSplitCell<<<blocks, gpuThreads>>>(verticesCellsNum.y, refinedVertices.data, refinedCells.data, refinedCellMeasures.data, originalCells.data,
                refinedVerticesCellsNum, tempVertices.data, tempCells.data, tempCellMeasures.data, tempOriginalCells.data);

        cudaDeviceSynchronize();
        copy_d2h(refinedVerticesCellsNum, &verticesCellsNum, 1);
    }

    //update tasks
    int *taskCount;
    int hostTaskCount;
    allocate_device(&taskCount, 1);

    int cellBlocks = blocksForSize(verticesCellsNum.y, gpuThreads2D);
    dim3 threads(gpuThreads2D, gpuThreads2D);

    if(simpleNeighborsTasks.size){
        zero_value_device(taskCount, 1);
        int taskBlocks = blocksForSize(simpleNeighborsTasks.size, gpuThreads2D);        

        dim3 blocks(taskBlocks, cellBlocks);
        kCountOrCreateTasks<<<blocks, threads>>>(simpleNeighborsTasks.size, verticesCellsNum.y, taskCount, simpleNeighborsTasks.data, originalCells.data);

        cudaDeviceSynchronize();
        copy_d2h(taskCount, &hostTaskCount, 1);

        refinedSimpleNeighborsTasks.allocate(hostTaskCount);
        d_simpleNeighborsResults.allocate(hostTaskCount);
        zero_value_device(taskCount, 1);
        kCountOrCreateTasks<<<blocks, threads>>>(simpleNeighborsTasks.size, verticesCellsNum.y, taskCount, simpleNeighborsTasks.data, originalCells.data, refinedSimpleNeighborsTasks.data);
    }

    if(attachedNeighborsTasks.size){
        zero_value_device(taskCount, 1);
        int taskBlocks = blocksForSize(attachedNeighborsTasks.size, gpuThreads2D);
        
        dim3 blocks(taskBlocks, cellBlocks);
        kCountOrCreateTasks<<<blocks, threads>>>(attachedNeighborsTasks.size, verticesCellsNum.y, taskCount, attachedNeighborsTasks.data, originalCells.data);

        cudaDeviceSynchronize();
        copy_d2h(taskCount, &hostTaskCount, 1);

        refinedAttachedNeighborsTasks.allocate(hostTaskCount);
        d_attachedNeighborsResults.allocate(hostTaskCount);
        zero_value_device(taskCount, 1);
        kCountOrCreateTasks<<<blocks, threads>>>(attachedNeighborsTasks.size, verticesCellsNum.y, taskCount, attachedNeighborsTasks.data, originalCells.data, refinedAttachedNeighborsTasks.data);
    }

    if(notNeighborsTasks.size){
        zero_value_device(taskCount, 1);
        int taskBlocks = blocksForSize(notNeighborsTasks.size, gpuThreads2D);
        
        dim3 blocks(taskBlocks, cellBlocks);
        kCountOrCreateTasks<<<blocks, threads>>>(notNeighborsTasks.size, verticesCellsNum.y, taskCount, notNeighborsTasks.data, originalCells.data);

        cudaDeviceSynchronize();
        copy_d2h(taskCount, &hostTaskCount, 1);

        refinedNotNeighborsTasks.allocate(hostTaskCount);
        d_notNeighborsResults.allocate(hostTaskCount);
        zero_value_device(taskCount, 1);
        kCountOrCreateTasks<<<blocks, threads>>>(notNeighborsTasks.size, verticesCellsNum.y, taskCount, notNeighborsTasks.data, originalCells.data, refinedNotNeighborsTasks.data);
    }

    printf("Refined mesh contains %d vertices and %d cells. Number of tasks: simple neighbors - %d, attached neighbors - %d, non-neighbors - %d\n",
            verticesCellsNum.x, verticesCellsNum.y, refinedSimpleNeighborsTasks.size, refinedAttachedNeighborsTasks.size, refinedNotNeighborsTasks.size);

    free_device(refinedVerticesCellsNum);
    free_device(taskCount);
}

void NumericalIntegrator3D::gatherResults(deviceVector<double4> &results, neighbour_type_enum neighborType) const
{
    int3 *refinedTasks;
    int refinedTasksSize;
    double4 *refinedResults;
    
    switch (neighborType)
    {
    case neighbour_type_enum::simple_neighbors:
        refinedTasks = refinedSimpleNeighborsTasks.data;
        refinedTasksSize = refinedSimpleNeighborsTasks.size;
        refinedResults = d_simpleNeighborsResults.data;
        break;
    case neighbour_type_enum::attached_neighbors:
        refinedTasks = refinedAttachedNeighborsTasks.data;
        refinedTasksSize = refinedAttachedNeighborsTasks.size;
        refinedResults = d_attachedNeighborsResults.data;
        break;
    case neighbour_type_enum::not_neighbors:
        refinedTasks = refinedNotNeighborsTasks.data;
        refinedTasksSize = refinedNotNeighborsTasks.size;
        refinedResults = d_notNeighborsResults.data;
        break;
    }

    if(results.size){
        int blocks = blocksForSize(refinedTasksSize);
        kSumIntegrationResults<<<blocks, gpuThreads>>>(refinedTasksSize, results.data, refinedResults, refinedTasks);
    }
}
