#include "cuda_helper.cuh"

__global__ void kFillOrdinal(int n, int *indices)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        indices[idx] = idx;
}

__global__ void kExtractIndices(int n, int *indices, int *counter, const unsigned char *mask)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        if(mask[idx]){
            int pos = atomicAdd(counter, 1);
            indices[pos] = idx;
        }
}

size_t requestFreeDeviceMemoryAmount()
{
    size_t freeMemory, totalMemory;
    checkCudaErrors(cudaMemGetInfo(&freeMemory, &totalMemory));
    printf("GPU memory usage: %5.1f MBytes free out of total %5.1f MBytes\n", freeMemory / 1048576.0f, totalMemory / 1048576.0f);

    return freeMemory;
}
