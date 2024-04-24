/*!
 * @file NumericalIntegrator3d.cu
 * @brief Implementation of the procedures of numerical integration, operations before (preparation of mesh and tasks) and after it (results summation, etc.)
 */
#include "NumericalIntegrator3d.cuh"

#include "common/cuda_memory.cuh"

__constant__ Point3 c_GaussPointsCoordinates[CONSTANTS::MAX_GAUSS_POINTS];      //!< Array of Gauss points coordinates (for use in the quadrature formula)
__constant__ double c_GaussPointsWeights[CONSTANTS::MAX_GAUSS_POINTS];          //!< Array of Gauss points weights (for use in the quadrature formula)
__constant__ int    c_GaussPointsNumber;                                        //!< Number of Gauss points

/*!
 * @brief Split an existing triangular cell into 4 new ones using midpoints of its edges
 * 
 * @param n Number of cells in the current mesh (may be original or previous iteration of refinement)
 * @param refinedVertices Vector of new vertices to be filled (containts vertices from the previous iteration of refinement)
 * @param refinedCells Vector of new cells to be filled
 * @param refinedCellMeasures Vector of new cell measures (areas)
 * @param originalCells Indices of original cells corresponding to the new cells
 * @param refinedVerticesCellsNum Counters for refined vertices (.x) and cells (.y)
 * @param tempVertices Vector of existing vertices used for refinement
 * @param tempCells Vector of existing cells used for refinement
 * @param tempCellMeasures Vector of measures (areas) for existing cells
 * @param tempOriginalCells Indices of original cells for the existing cells
 * @param refinedCellParents Vector of direct parents for the new refined cells
 * @param cellRequiresRefinement Vector of flags showing whether the original cell requiresrefinement
 * 
 * Process of splitting a single cell is organized as the following:
 * - for each cell in the current mesh we check whether it belongs to an original cells which needs refinement (or the whole mesh is refined);
 * - 3 new vertices are created in the midpoints of edges of the existing triangle and added to the vector of new vertices;
 * - 4 new cells are created and added to the vector of new cells. Counters for numbers of new vertices and cells are increased using atomicAdd;
 * - measures are filled with the value of \f$\frac14 S_i\f$ (\f$S_i\f$ is the area of triangle being split);
 * - index of the original cell is directly copied to the new cells from the cell being refined;
 * - if needed, index of the direct parent cell of the new cells is assigned to the index of cell being refined.
 */
__global__ void kSplitCell(int n, Point3 *refinedVertices, int3 *refinedCells, double *refinedCellMeasures, int *originalCells, int2 *refinedVerticesCellsNum,
        const Point3 *tempVertices, const int3 *tempCells, const double *tempCellMeasures, const int *tempOriginalCells, int *refinedCellParents = nullptr, const unsigned char *cellRequiresRefinement = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int originalCellIndex = tempOriginalCells[idx];

        if(cellRequiresRefinement && !cellRequiresRefinement[originalCellIndex])
            return;
        
        //data of the triangle to be split
        const int3 triangle = tempCells[idx];
        const Point3 triA = tempVertices[triangle.x];
        const Point3 triB = tempVertices[triangle.y];
        const Point3 triC = tempVertices[triangle.z];
        const double measure = tempCellMeasures[idx];

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

        if(refinedCellParents){
            refinedCellParents[cellIndex] = idx;
            refinedCellParents[cellIndex + 1] = idx;
            refinedCellParents[cellIndex + 2] = idx;
            refinedCellParents[cellIndex + 3] = idx;
        }
    }
}

/*!
 * @brief Count and (optionally) create new integration tasks for the refined cells from the existing ones
 * 
 * @param tasksNum Number of the existing integration tasks
 * @param refinedCellsNum Number of cells in the refined mesh (for which the tasks are created/counted)
 * @param counter Counter of the new tasks
 * @param tasks Existing integration tasks from which new ones are created
 * @param originalCells Indices of cell of original mesh for the refined mesh cells
 * @param tempOriginalCells Indices of cell of original mesh for the previous mesh cells
 * @param refinedCellParents Indices of direct parents of the cells of the refined mesh
 * @param taskConverged Flags showing that the original integration task has converged
 * @param refinedTasks Vector of new tasks being created (if nullptr then the tasks are only counted)
 * 
 * Existing tasks are being analyzed in large blocks, indices of original cells for the new refined cells
 * (and optionally indices of direct cell parents for them) are loaded into shared memory.
 * After the indices of original cells have been loaded, we analyze each existing integration task. If tasks are created for all cells
 * or this task is the part of original task has not converged we proceed further and check loaded indices of original cells for new ones.
 * If
 * -# These indices are the same for the \f$i\f$-th panel of the existing task and the new refined cell and
 * -# The new refined cell has the \f$i\f$-th panel of the existing task as its direct parent
 * then a new task is counted and (optionally) created for this new refined cell and the \f$j\f$-th panel of the existing task.
 */
__global__ void kCountOrCreateTasks(int tasksNum, int refinedCellsNum, int *counter, const int3 *tasks, const int *originalCells, const int *tempOriginalCells, const int *refinedCellParents = nullptr, unsigned char *taskConverged = nullptr, int3 *refinedTasks = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int originalCellsSh[gpuThreadsMax];
    __shared__ int cellParentsSh[gpuThreadsMax];

	//do not exit the kernel immediately when idx >= tasksNum
	//because the thread might be required to perform loading into shared memory
	for(int blockStart = 0; blockStart < refinedCellsNum; blockStart += gpuThreadsMax){
		if(blockStart + threadIdx.x < refinedCellsNum){
			originalCellsSh[threadIdx.x] = originalCells[blockStart + threadIdx.x];
			if(refinedCellParents)
				cellParentsSh[threadIdx.x] = refinedCellParents[blockStart + threadIdx.x];
		}
		__syncthreads();

		if(idx < tasksNum){
			const int3 oldTask = tasks[idx];
            const int taskOriginalCellI = tempOriginalCells[oldTask.x];

			if(!taskConverged || !taskConverged[oldTask.z])
				for(int cell = 0; cell < gpuThreadsMax; ++cell)
					if(blockStart + cell < refinedCellsNum){
						const int originalCell = originalCellsSh[cell];

						if(taskOriginalCellI == originalCell && (!refinedCellParents || cellParentsSh[cell] == oldTask.x)){
							int pos = atomicAdd(counter, 1);

							if(refinedTasks)
								refinedTasks[pos] = { blockStart + cell, oldTask.y, oldTask.z };
						}
					}
		}

		__syncthreads();
	}
}

/*!
 * @brief Kernel function for summation of results of integration over refined cells
 * 
 * @param n Number of tasks of integration over refined cells
 * @param results Target vector of integration results for original tasks
 * @param refinedResults Vector of integration results for refined tasks
 * @param refinedTasks Vector of integration tasks over refined cells
 *
 * Summation is performed using atomicAdd for double-precision numbers, separately for each component of \f$\mathbf{\Psi}\f$ and \f$\theta\f$
 */
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

/*!
 * @brief Extracting the set of original cells that require refinement (by setting the flag for them to 'true') 
 * 
 * @param n Number of integration tasks which have not converged
 * @param cellNeedsRefinement Vector of flags showing whether an original cell requires further refinement
 * @param restTasks Indices of integration tasks over refined cells which have not converged yet
 * @param tasks Vector of integration tasks for refined cells
 * 
 * Remaining integration tasks for refined cells are analyzed and their 3rd indices are used to determine the original cell
 * that needs further refinement. If the cell is the \f$i\f$-th control panel for at least one task its flag is set to 'true'.
 */
__global__ void kExtractCellNeedsRefinement(int n, unsigned char *cellNeedsRefinement, const int *restTasks, const int3 *tasks)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int taskIndex = restTasks[idx];
        const int3 originalTask = tasks[taskIndex];

        cellNeedsRefinement[originalTask.x] = true;
    }
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
    copy_h2const(&GaussPointsNum, &c_GaussPointsNumber, 1);

    errorControlType = error_control_type_enum::automatic_error_control;

    allocate_device(&refinedVerticesCellsNum, 1);
    allocate_device(&taskCount, 1);
}

NumericalIntegrator3D::~NumericalIntegrator3D()
{
    free_device(refinedVerticesCellsNum);
    free_device(taskCount);
    free_device(d_cellsToBeRefinedCount);
}

void NumericalIntegrator3D::setFixedRefinementLevel(int refinementLevel)
{
    errorControlType = error_control_type_enum::fixed_refinement_level;
    meshRefinementLevel = refinementLevel;
}

void NumericalIntegrator3D::prepareTasksAndMesh(const deviceVector<int3> &simpleNeighborsTasks, const deviceVector<int3> &attachedNeighborsTasks, const deviceVector<int3> &notNeighborsTasks)
{
    verticesCellsNum = { mesh.getVertices().size, mesh.getCells().size };

    int refinedVerticesNum, refinedCellsNum;
    if(errorControlType == error_control_type_enum::automatic_error_control){
        refinedVerticesNum = verticesCellsNum.x + verticesCellsNum.y * ((1 << (2 * CONSTANTS::MAX_REFINE_LEVEL)) - 1);
        refinedCellsNum = (1 << (2 * CONSTANTS::MAX_REFINE_LEVEL)) * verticesCellsNum.y;
    } else if(meshRefinementLevel){
        refinedVerticesNum = verticesCellsNum.x + verticesCellsNum.y * ((1 << (2 * meshRefinementLevel)) - 1);
        refinedCellsNum = (1 << (2 * meshRefinementLevel)) * verticesCellsNum.y;
    } else {
        refinedVerticesNum = verticesCellsNum.x;
        refinedCellsNum = verticesCellsNum.y;
    }

    refinedVertices.allocate(refinedVerticesNum);
    refinedCells.allocate(refinedCellsNum);
    refinedCellMeasures.allocate(refinedCellsNum);

    //initialize vectors using original data (in case of refinement it will then be moved to the temporary buffers)
    copy_d2d(mesh.getVertices().data, refinedVertices.data, verticesCellsNum.x);
    copy_d2d(mesh.getCells().data, refinedCells.data, verticesCellsNum.y);
    copy_d2d(mesh.getCellMeasures().data, refinedCellMeasures.data, verticesCellsNum.y);

    if(errorControlType == error_control_type_enum::automatic_error_control || !meshRefinementLevel){
        if(simpleNeighborsTasks.size){
            int taskSize = simpleNeighborsTasks.size;
            if(errorControlType == error_control_type_enum::automatic_error_control){
                simpleNeighborsIntegralsConverged.allocate(taskSize);
                zero_value_device(simpleNeighborsIntegralsConverged.data, taskSize);
                taskSize *= CONSTANTS::MAX_AUTO_REFINEMENT_TASK_COEFFICIENT;
            }

            refinedSimpleNeighborsTasks.allocate(taskSize);
            copy_d2d(simpleNeighborsTasks.data, refinedSimpleNeighborsTasks.data, simpleNeighborsTasks.size);
            refinedSimpleNeighborsTasks.resize(simpleNeighborsTasks.size);

            d_simpleNeighborsResults.allocate(taskSize);
            if(errorControlType == error_control_type_enum::automatic_error_control)
                tempRefinedSimpleNeighborsTasks.allocate(taskSize);
        }

        if(attachedNeighborsTasks.size){
            int taskSize = attachedNeighborsTasks.size;
            if(errorControlType == error_control_type_enum::automatic_error_control){
                attachedNeighborsIntegralsConverged.allocate(taskSize);
                zero_value_device(attachedNeighborsIntegralsConverged.data, taskSize);
                taskSize *= CONSTANTS::MAX_AUTO_REFINEMENT_TASK_COEFFICIENT;
            }

            refinedAttachedNeighborsTasks.allocate(taskSize);
            copy_d2d(attachedNeighborsTasks.data, refinedAttachedNeighborsTasks.data, attachedNeighborsTasks.size);
            refinedAttachedNeighborsTasks.resize(attachedNeighborsTasks.size);

            d_attachedNeighborsResults.allocate(taskSize);
            if(errorControlType == error_control_type_enum::automatic_error_control)
                tempRefinedAttachedNeighborsTasks.allocate(taskSize);
        }

        if(notNeighborsTasks.size){
            int taskSize = notNeighborsTasks.size;
            if(errorControlType == error_control_type_enum::automatic_error_control){
                notNeighborsIntegralsConverged.allocate(taskSize);
                zero_value_device(notNeighborsIntegralsConverged.data, taskSize);
                taskSize *= CONSTANTS::MAX_AUTO_REFINEMENT_TASK_COEFFICIENT;
            }

            refinedNotNeighborsTasks.allocate(taskSize);
            copy_d2d(notNeighborsTasks.data, refinedNotNeighborsTasks.data, notNeighborsTasks.size);
            refinedNotNeighborsTasks.resize(notNeighborsTasks.size);

            d_notNeighborsResults.allocate(taskSize);
            if(errorControlType == error_control_type_enum::automatic_error_control)
                tempRefinedNotNeighborsTasks.allocate(taskSize);
        }

        if(errorControlType == error_control_type_enum::fixed_refinement_level)
            return;
    }

    tempVertices.allocate(refinedVerticesNum);
    tempCells.allocate(refinedCellsNum);
    tempCellMeasures.allocate(refinedCellsNum);

    originalCells.allocate(refinedCellsNum);
    tempOriginalCells.allocate(refinedCellsNum);

    //initialize number of indices of original cells with 0,1,2,...,ncells-1
    unsigned int blocks = blocksForSize(verticesCellsNum.y);
    kFillOrdinal<<<blocks, gpuThreads>>>(verticesCellsNum.y, originalCells.data);

    copy_h2d(&verticesCellsNum, refinedVerticesCellsNum, 1);

    if(errorControlType == error_control_type_enum::automatic_error_control){
        refinedCellParents.allocate(refinedCellsNum);
        cellsToBeRefined.allocate(refinedCellsNum);
        cellRequiresRefinement.allocate(verticesCellsNum.y);
        allocate_device(&d_cellsToBeRefinedCount, 1);

        //initialize vectors containing numbers of refinement iterations for each cell (for different types of neighbor)
        simpleNeighborsRefinementsRequired.allocate(verticesCellsNum.y);
        zero_value_device(simpleNeighborsRefinementsRequired.data, verticesCellsNum.y);
        attachedNeighborsRefinementsRequired.allocate(verticesCellsNum.y);
        zero_value_device(attachedNeighborsRefinementsRequired.data, verticesCellsNum.y);
        notNeighborsRefinementsRequired.allocate(verticesCellsNum.y);
        zero_value_device(notNeighborsRefinementsRequired.data, verticesCellsNum.y);

        return;
    }

    //case of full mesh refinement
    for(int i = 0; i < meshRefinementLevel; ++i)
        refineMesh();

    //update tasks
    int hostTaskCount;

    if(simpleNeighborsTasks.size){
        hostTaskCount = updateTasks(simpleNeighborsTasks, neighbour_type_enum::simple_neighbors);
        d_simpleNeighborsResults.allocate(hostTaskCount);
    }

    if(attachedNeighborsTasks.size){
        hostTaskCount = updateTasks(attachedNeighborsTasks, neighbour_type_enum::attached_neighbors);
        d_attachedNeighborsResults.allocate(hostTaskCount);
    }

    if(notNeighborsTasks.size){
        hostTaskCount = updateTasks(notNeighborsTasks, neighbour_type_enum::not_neighbors);
        d_notNeighborsResults.allocate(hostTaskCount);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    printf("Refined mesh contains %d vertices and %d cells. Number of tasks: simple neighbors - %d, attached neighbors - %d, non-neighbors - %d\n",
            verticesCellsNum.x, verticesCellsNum.y, refinedSimpleNeighborsTasks.size, refinedAttachedNeighborsTasks.size, refinedNotNeighborsTasks.size);
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

void NumericalIntegrator3D::refineMesh(neighbour_type_enum updateTasksNeighborType)
{
    std::swap(refinedVertices.data, tempVertices.data);
    std::swap(refinedCells.data, tempCells.data);
    std::swap(refinedCellMeasures.data, tempCellMeasures.data);
    std::swap(originalCells.data, tempOriginalCells.data);

    //current vertex list forms the first part of the new vertex list
    //leave the vertex count equal to the size of the current vertex list and set the cell count to zero
    copy_d2d(tempVertices.data, refinedVertices.data, verticesCellsNum.x);
    zero_value_device((int*)refinedVerticesCellsNum + 1, 1);

    const bool refineWholeMesh = errorControlType == error_control_type_enum::fixed_refinement_level;
    unsigned int blocks = blocksForSize(verticesCellsNum.y);

    kSplitCell<<<blocks, gpuThreads>>>(verticesCellsNum.y, refinedVertices.data, refinedCells.data, refinedCellMeasures.data, originalCells.data,
            refinedVerticesCellsNum, tempVertices.data, tempCells.data, tempCellMeasures.data, tempOriginalCells.data, refineWholeMesh ? nullptr : refinedCellParents.data, refineWholeMesh ? nullptr : cellRequiresRefinement.data);

    checkCudaErrors(cudaDeviceSynchronize());
    copy_d2h(refinedVerticesCellsNum, &verticesCellsNum, 1);

    //in case of automatic error control update the tasks
    if(errorControlType == error_control_type_enum::automatic_error_control){
        deviceVector<int3> *refinedTasks, *tempRefinedTasks;
        switch (updateTasksNeighborType)
        {
        case neighbour_type_enum::simple_neighbors:
            refinedTasks = &refinedSimpleNeighborsTasks;
            tempRefinedTasks = &tempRefinedSimpleNeighborsTasks;
            break;
        case neighbour_type_enum::attached_neighbors:
            refinedTasks = &refinedAttachedNeighborsTasks;
            tempRefinedTasks = &tempRefinedAttachedNeighborsTasks;
            break;
        case neighbour_type_enum::not_neighbors:
            refinedTasks = &refinedNotNeighborsTasks;
            tempRefinedTasks = &tempRefinedNotNeighborsTasks;
            break;
        default:
            return;
        }

        refinedTasks->swap(*tempRefinedTasks);
        updateTasks(*tempRefinedTasks, updateTasksNeighborType);
    }
}

void NumericalIntegrator3D::resetMesh()
{
    verticesCellsNum = { mesh.getVertices().size, mesh.getCells().size };

    copy_d2d(mesh.getVertices().data, refinedVertices.data, verticesCellsNum.x);
    copy_d2d(mesh.getCells().data, refinedCells.data, verticesCellsNum.y);
    copy_d2d(mesh.getCellMeasures().data, refinedCellMeasures.data, verticesCellsNum.y);

    unsigned int blocks = blocksForSize(verticesCellsNum.y);
    kFillOrdinal<<<blocks, gpuThreads>>>(verticesCellsNum.y, originalCells.data);

    copy_h2d(&verticesCellsNum, refinedVerticesCellsNum, 1);
}

int NumericalIntegrator3D::determineCellsToBeRefined(const deviceVector<int> &restTasks, const deviceVector<int3> *tasks, neighbour_type_enum neighborType)
{
    deviceVector<unsigned char> *refinementsRequired;

    switch (neighborType)
    {
    case neighbour_type_enum::simple_neighbors:
        refinementsRequired = &simpleNeighborsRefinementsRequired;
        break;
    case neighbour_type_enum::attached_neighbors:
        refinementsRequired = &attachedNeighborsRefinementsRequired;
        break;
    case neighbour_type_enum::not_neighbors:
        refinementsRequired = &notNeighborsRefinementsRequired;
        break;
    default:
        return 0;
    }

    zero_value_device(cellRequiresRefinement.data, cellRequiresRefinement.size);
    unsigned int blocks = blocksForSize(restTasks.size);
    kExtractCellNeedsRefinement<<<blocks, gpuThreads>>>(restTasks.size, cellRequiresRefinement.data, restTasks.data, tasks->data);

    zero_value_device(d_cellsToBeRefinedCount, 1);
    blocks = blocksForSize(cellRequiresRefinement.size);
    kExtractIndices<<<blocks, gpuThreads>>>(cellRequiresRefinement.size, cellsToBeRefined.data, d_cellsToBeRefinedCount, cellRequiresRefinement.data);

    checkCudaErrors(cudaDeviceSynchronize());

    int cellsToBeRefinedCount;
    copy_d2h(d_cellsToBeRefinedCount, &cellsToBeRefinedCount, 1);

    blocks = blocksForSize(cellsToBeRefinedCount);
    kIncreaseValue<unsigned char><<<blocks, gpuThreads>>>(cellsToBeRefinedCount, refinementsRequired->data, 1, cellsToBeRefined.data);

    cellsToBeRefined.resize(cellsToBeRefinedCount);

    return cellsToBeRefinedCount;
}

int NumericalIntegrator3D::updateTasks(const deviceVector<int3> &originalTasks, neighbour_type_enum neighborType)
{
    deviceVector<int3> *refinedTasks;
    unsigned char *integralsConverged = getIntegralsConverged(neighborType)->data;

    switch (neighborType)
    {
    case neighbour_type_enum::simple_neighbors:
        refinedTasks = &refinedSimpleNeighborsTasks;
        break;
    case neighbour_type_enum::attached_neighbors:
        refinedTasks = &refinedAttachedNeighborsTasks;
        break;
    case neighbour_type_enum::not_neighbors:
        refinedTasks = &refinedNotNeighborsTasks;
        break;
    }

    int hostTaskCount;
    
    zero_value_device(taskCount, 1);
    unsigned int blocks = blocksForSize(originalTasks.size, gpuThreadsMax);
    
    kCountOrCreateTasks<<<blocks, gpuThreadsMax>>>(originalTasks.size, verticesCellsNum.y, taskCount, originalTasks.data, originalCells.data, tempOriginalCells.data, refinedCellParents.data, integralsConverged);

    checkCudaErrors(cudaDeviceSynchronize());
    copy_d2h(taskCount, &hostTaskCount, 1);

    if(refinedTasks->data){
        refinedTasks->resize(hostTaskCount);

        deviceVector<double4> *results;
        switch (neighborType)
        {
        case neighbour_type_enum::simple_neighbors:
            results = &d_simpleNeighborsResults;
            break;
        case neighbour_type_enum::attached_neighbors:
            results = &d_attachedNeighborsResults;
            break;
        case neighbour_type_enum::not_neighbors:
            results = &d_notNeighborsResults;
            break;
        default:
            results = nullptr;
            break;
        }

        if(results)
            results->resize(hostTaskCount);
    } else
        refinedTasks->allocate(hostTaskCount);

    zero_value_device(taskCount, 1);
    kCountOrCreateTasks<<<blocks, gpuThreadsMax>>>(originalTasks.size, verticesCellsNum.y, taskCount, originalTasks.data, originalCells.data, tempOriginalCells.data, refinedCellParents.data, integralsConverged, refinedTasks->data);

    return hostTaskCount;
}

__device__ double4 integrate4D(const double4 *functionValues)
{
    double4 res = { 0.0, 0.0, 0.0, 0.0 };
    for(int i = 0; i < c_GaussPointsNumber; ++i)
        res += c_GaussPointsWeights[i] * functionValues[i];
    
    return res;
}

__device__ void calculateQuadraturePoints(Point3 *quadraturePoints, const Point3 *vertices, const int3 &triangle)
{
    Point3 triangleVertices[3];
    triangleVertices[0] = vertices[triangle.x];
    triangleVertices[1] = vertices[triangle.y];
    triangleVertices[2] = vertices[triangle.z];

    for(int i = 0; i < c_GaussPointsNumber; ++i){
        Point3 res = { 0.0, 0.0, 0.0 };
        const Point3 Lcoordinates = c_GaussPointsCoordinates[i];

        for(int j = 0; j < 3; ++j)
            res += *(&Lcoordinates.x + j) * triangleVertices[j];

        quadraturePoints[i] = res;
    }
}
