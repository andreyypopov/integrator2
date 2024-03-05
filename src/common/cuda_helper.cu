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