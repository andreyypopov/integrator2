#include "Mesh3d.cuh"

#include "common/cuda_memory.cuh"
#include "common/constants.h"

#include <fstream>
#include <vector>

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
        centers[idx] = (vertices[triangle.x] + vertices[triangle.y] + vertices[triangle.z]) * CONSTANTS::ONE_THIRD;
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

__global__ void kDetermineNeighborType(int n, const int3 *cells, int2 *simpleNeighbors, int *simpleNeighborsNum,
    int2 *attachedNeighbors, int *attachedNeighborsNum){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tri1idx = idx / blockDim.x;
    int tri2idx = idx % blockDim.x;

    if(tri1idx < n && tri2idx < n && tri1idx < tri2idx){
        unsigned int commonPoints = 0;
        int3 tri1 = cells[tri1idx];
        int3 tri2 = cells[tri2idx];

        if(tri1.x == tri2.x || tri1.x == tri2.y || tri1.x == tri2.z)
            ++commonPoints;

        if(tri1.y == tri2.x || tri1.y == tri2.y || tri1.y == tri2.z)
            ++commonPoints;

        if(tri1.z == tri2.x || tri1.z == tri2.y || tri1.z == tri2.z)
            ++commonPoints;
        
        if(commonPoints == 1){
            int pos = atomicAdd(simpleNeighborsNum, 1);
            simpleNeighbors[pos] = int2({ tri1idx, tri2idx });
        }

        if(commonPoints == 2){
            int pos = atomicAdd(attachedNeighborsNum, 1);
            attachedNeighbors[pos] = int2({ tri1idx, tri2idx });
        }
    }
}

Mesh3D::~Mesh3D(){
    if(verticesNum)
        free_device(vertices);

    if(cellsNum){
        free_device(cells);
        free_device(cellNormals);
        free_device(cellCenters);
        free_device(cellMeasures);
    }

    if(quadraturePointsNum)
        free_device(quadraturePoints);

    if(simpleNeighborsNum){
        free_device(d_simpleNeighborsNum);
        free_device(simpleNeighbors);
    }

    if(attachedNeighborsNum){
        free_device(d_attachedNeighborsNum);
        free_device(attachedNeighbors);
    }
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
            hostVertices.push_back(vertex * scale);
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

        verticesNum = numVertices;
        cellsNum = numCells;

        allocate_device(&vertices, verticesNum);
        allocate_device(&cells, cellsNum);
        allocate_device(&cellNormals, cellsNum);
        allocate_device(&cellCenters, cellsNum);
        allocate_device(&cellMeasures, cellsNum);

        copy_h2d(hostVertices.data(), vertices, verticesNum);
        copy_h2d(hostCells.data(), cells, cellsNum);

        printf("Loaded mesh with %d vertices and %d cells\n", verticesNum, cellsNum);

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
    unsigned int blocks = blocksForSize(cellsNum);
    kCalculateCellNormal<<<blocks, gpuThreads>>>(cellsNum, vertices, cells, cellNormals);
}

void Mesh3D::calculateCenters(){
    unsigned int blocks = blocksForSize(cellsNum);
    kCalculateCellCenter<<<blocks, gpuThreads>>>(cellsNum, vertices, cells, cellCenters);
}

void Mesh3D::calculateMeasures(){
    unsigned int blocks = blocksForSize(cellsNum);
    kCalculateCellMeasure<<<blocks, gpuThreads>>>(cellsNum, vertices, cells, cellMeasures);
}

void Mesh3D::fillNeightborsLists(){
    allocate_device(&d_simpleNeighborsNum, 1);
    allocate_device(&d_attachedNeighborsNum, 1);
    zero_value_device(d_simpleNeighborsNum, 1);
    zero_value_device(d_attachedNeighborsNum, 1);

    allocate_device(&simpleNeighbors, cellsNum * CONSTANTS::MAX_SIMPLE_NEIGHBORS_PER_VERTEX / 2);    // number of triangles * (3 neighbors * 3 vertices per triangle) / 2 (discard duplicated pairs)
    allocate_device(&attachedNeighbors, cellsNum * 3 / 2);  // number of triangles * (3 vertices edges per triangle) / 2 (discard duplicated pairs)

    unsigned int blocks = blocksForSize(cellsNum * cellsNum);
    kDetermineNeighborType<<<blocks, gpuThreads>>>(cellsNum, cells, simpleNeighbors, d_simpleNeighborsNum, attachedNeighbors, d_attachedNeighborsNum);
    copy_d2h(d_simpleNeighborsNum, &simpleNeighborsNum, 1);
    copy_d2h(d_attachedNeighborsNum, &attachedNeighborsNum, 1);

    printf("Found %d simple neighbors and %d attached neighbors\n", simpleNeighborsNum, attachedNeighborsNum);
}
