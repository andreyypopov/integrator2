#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <stdio.h>

const int gpuThreads = 256;
const int gpuThreadsMax = 1024;
const int gpuThreads2D = 16;

inline unsigned int blocksForSize(unsigned int n, unsigned int maxThreads = gpuThreads){
    return (n + maxThreads - 1) / maxThreads;
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n", file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void kFillOrdinal(int n, int *indices);

__global__ void kExtractIndices(int n, int *indices, int *counter, const unsigned char *mask);

template<class T>
__global__ void kFillValue(int n, T *array, T value, int *indices = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        if(indices)
            idx = indices[idx];
        
        array[idx] = value;
    }
}

template<class T>
__global__ void kIncreaseValue(int n, T *array, T value, int *indices = nullptr)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        if(indices)
            idx = indices[idx];

        array[idx] += value;
    }
}

#endif // CUDA_HELPER_CUH