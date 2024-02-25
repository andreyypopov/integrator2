#include "Mesh3d.cuh"

#include "common/cuda_memory.cuh"
#include "common/constants.h"

#include <fstream>

__global__ void kCalculateCellNormal(int n, const Point3 *vertices, const int3 *cells, Point3 *normals){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 triangle = cells[idx];
        const Point3 v12 = vertices[triangle.y] - vertices[triangle.x];
        const Point3 v13 = vertices[triangle.z] - vertices[triangle.x];

        normals[idx] = normalize(cross(v12, v13)); 
    }
}

__global__ void kCalculateCellCenter(int n, const Point3 *vertices, const int3 *cells, Point3 *centers){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 triangle = cells[idx];
        centers[idx] = CONSTANTS::ONE_THIRD * (vertices[triangle.x] + vertices[triangle.y] + vertices[triangle.z]);
    }
}

__global__ void kCalculateCellMeasure(int n, const Point3 *vertices, const int3 *cells, double *measures){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 triangle = cells[idx];
        const Point3 v12 = vertices[triangle.y] - vertices[triangle.x];
        const Point3 v13 = vertices[triangle.z] - vertices[triangle.x];

        measures[idx] = vector_length(cross(v12, v13)) * 0.5; 
    }
}

__global__ void kDetermineNeighborType(int n, const int3 *cells, int3 *simpleNeighbors, int *simpleNeighborsNum,
    int3 *attachedNeighbors, int *attachedNeighborsNum, int3 *notNeighbors, int *notNeighborsNum){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int3 sharedCells[gpuThreadsMax];

	//do not exit the kernel immediately when idx >= n
	//because the thread might be required to perform loading into shared memory  
	for(int blockStart = 0; blockStart < n; blockStart += gpuThreadsMax){
		if(blockStart + threadIdx.x < n)
			sharedCells[threadIdx.x] = cells[blockStart + threadIdx.x];
		__syncthreads();

		if(idx < n){
			const int3 tri1 = cells[idx];

			for(int cell = 0; cell < gpuThreadsMax; ++cell)
				if(idx < blockStart + cell && blockStart + cell < n){
					unsigned int commonPoints = 0;
			
					const int3 tri2 = sharedCells[cell];

					if(tri1.x == tri2.x || tri1.x == tri2.y || tri1.x == tri2.z)
						++commonPoints;

					if(tri1.y == tri2.x || tri1.y == tri2.y || tri1.y == tri2.z)
						++commonPoints;

					if(tri1.z == tri2.x || tri1.z == tri2.y || tri1.z == tri2.z)
						++commonPoints;
					
					if(commonPoints == 0){
						int pos = atomicAdd(notNeighborsNum, 1);
						notNeighbors[pos] = int3({ (int)idx, blockStart + cell, pos });
					}
					
					if(commonPoints == 1){
						int pos = atomicAdd(simpleNeighborsNum, 1);
						simpleNeighbors[pos] = int3({ (int)idx, blockStart + cell, pos });
					}

					if(commonPoints == 2){
						int pos = atomicAdd(attachedNeighborsNum, 1);
						attachedNeighbors[pos] = int3({ (int)idx, blockStart + cell, pos });
					}
				}
		}
		__syncthreads();
	}
}

Mesh3D::~Mesh3D(){
    free_device(d_simpleNeighborsNum);
    free_device(d_attachedNeighborsNum);
    free_device(d_notNeighborsNum);
}

bool Mesh3D::loadMeshFromFile(const std::string &filename, double scale)
{
    std::ifstream meshFile(filename);

    if(meshFile.is_open()){
        int numVertices, numCells;
        int tmp;

        meshFile >> numVertices >> numCells;

        std::vector<Point3> hostVertices;
        std::vector<int3> hostCells;

        hostVertices.reserve(numVertices);
        hostCells.reserve(numCells);

        for(int i = 0; i < numVertices; ++i){
            Point3 vertex;
            meshFile >> tmp >> vertex.x >> vertex.y >> vertex.z;
            hostVertices.push_back(scale * vertex);
        }

        while(!meshFile.eof()){
            meshFile >> tmp >> tmp;
            if(tmp == 203){ //encountered a triangle
                int3 triangle;
                meshFile >> triangle.x >> triangle.y >> triangle.z;
                
                //indices of vertices are base-1 in the imported files
                triangle.x -= 1;
                triangle.y -= 1;
                triangle.z -= 1;

                hostCells.push_back(triangle);
            } else {        //encountered an entity of another type
                numCells -= 1;
                meshFile >> tmp >> tmp;
            }
        }

        meshFile.close();

        vertices.allocate(numVertices);
        cells.allocate(numCells);
        cellNormals.allocate(numCells);
        cellCenters.allocate(numCells);
        cellMeasures.allocate(numCells);

        copy_h2d(hostVertices.data(), vertices.data, vertices.size);
        copy_h2d(hostCells.data(), cells.data, cells.size);

        printf("Loaded mesh with %d vertices and %d cells\n", numVertices, numCells);

        return true;
    } else {
        printf("Error while opening the file\n");
        return false;
    }
}

void Mesh3D::prepareMesh(){
    calculateNormals();
    calculateCenters();
    calculateMeasures();
    fillNeightborsLists();

    cudaDeviceSynchronize();
}

void Mesh3D::calculateNormals(){
    unsigned int blocks = blocksForSize(cells.size);
    kCalculateCellNormal<<<blocks, gpuThreads>>>(cells.size, vertices.data, cells.data, cellNormals.data);
}

void Mesh3D::calculateCenters(){
    unsigned int blocks = blocksForSize(cells.size);
    kCalculateCellCenter<<<blocks, gpuThreads>>>(cells.size, vertices.data, cells.data, cellCenters.data);
}

void Mesh3D::calculateMeasures(){
    unsigned int blocks = blocksForSize(cells.size);
    kCalculateCellMeasure<<<blocks, gpuThreads>>>(cells.size, vertices.data, cells.data, cellMeasures.data);
}

void Mesh3D::fillNeightborsLists(){
    allocate_device(&d_simpleNeighborsNum, 1);
    allocate_device(&d_attachedNeighborsNum, 1);
    allocate_device(&d_notNeighborsNum, 1);
    zero_value_device(d_simpleNeighborsNum, 1);
    zero_value_device(d_attachedNeighborsNum, 1);
    zero_value_device(d_notNeighborsNum, 1);

    simpleNeighbors.allocate(cells.size * CONSTANTS::MAX_SIMPLE_NEIGHBORS_PER_VERTEX / 2);    // number of triangles * (3 neighbors * 3 vertices per triangle) / 2 (discard duplicated pairs)
    attachedNeighbors.allocate(cells.size * 3 / 2);  // number of triangles * (3 vertices edges per triangle) / 2 (discard duplicated pairs)
    notNeighbors.allocate(cells.size * cells.size / 2);  // all pairs of triangls

    unsigned int blocks = blocksForSize(cells.size, gpuThreadsMax);
    kDetermineNeighborType<<<blocks, gpuThreadsMax>>>(cells.size, cells.data, simpleNeighbors.data, d_simpleNeighborsNum, attachedNeighbors.data, d_attachedNeighborsNum, notNeighbors.data, d_notNeighborsNum);
    copy_d2h(d_simpleNeighborsNum, &simpleNeighbors.size, 1);
    copy_d2h(d_attachedNeighborsNum, &attachedNeighbors.size, 1);
    copy_d2h(d_notNeighborsNum, &notNeighbors.size, 1);

    printf("Found %d pairs of simple neighbors and %d pairs of attached neighbors, %d pairs are not neighbors\n", simpleNeighbors.size, attachedNeighbors.size, notNeighbors.size);
}

void exportMeshToObj(const std::string &filename, const std::vector<Point3> &vertices, const std::vector<int3> &cells)
{
    std::ofstream outputFile(filename.c_str());

    for(const auto &pt : vertices)
        outputFile << "v " << pt.x << " " << pt.y << " " << pt.z << std::endl;

    for(const auto &triangle : cells)
        outputFile << "f " << triangle.x + 1 << " " << triangle.y + 1 << " " << triangle.z + 1 << std::endl;

    outputFile.close();
}
