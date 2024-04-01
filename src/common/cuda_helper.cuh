#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <stdio.h>

const int gpuThreads = 256;             //!< Standard CUDA block size for most part of the kernel functions
const int gpuThreadsMax = 1024;         //!< Increased size of the CUDA block for kernel functions that use shared memory
const int gpuThreads2D = 16;            //!< Decreased size of the CUDA block for kernel functions that use 2D geometry

/*!
 * @brief Calculate the size of CUDA grid for the size of the data to be processed
 * 
 * @param n Number of elements to be processed
 * @param maxThreads Number of threads in a block
 * @return Number of blocks in the grid
 */
inline unsigned int blocksForSize(unsigned int n, unsigned int maxThreads = gpuThreads){
    return (n + maxThreads - 1) / maxThreads;
}

/*!
 * @brief Function for checking CUDA errors
 * 
 * @tparam T Data type of the result of the function to be checked (cudaError_t)
 * @param result Result of the function execution
 * @param func Function name
 * @param file Name of the source file containg the function
 * @param line Line containing the function
 */
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

/*!
 * @brief Macro for CUDA error checks
 */
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

/*!
 * @brief Macro for asynchronous error checking
 */
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

/*!
 * @brief Function for asynchronous error checking
 * 
 * @param errorMessage Error message
 * @param file Name of the file which caused the error
 * @param line Line of the file which caused the error
 */
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

/*!
 * @brief Kernel function which fill the vector with values (0, n-1) 
 * 
 * @param n Number of elements in the vector
 * @param indices Raw pointer to the vector to be filled
 */
__global__ void kFillOrdinal(int n, int *indices);

/*!
 * @brief Kernel function for extraction of indices of mask array elements which are non-zero
 * 
 * @param n Number of elements in the mask array
 * @param indices Raw pointer to the target array with indices of non-zero element
 * @param counter Counter of the non-zero element
 * @param mask Mask array to extract indices from
 */
__global__ void kExtractIndices(int n, int *indices, int *counter, const unsigned char *mask);

/*!
 * @brief Kernel function to fill the elements of the vector (all or only with certain indices) with one specified value
 * 
 * @tparam T Data type
 * @param n Number of element to fill
 * @param array Raw pointer to the target array to be filled
 * @param value Value to be assigned to each element
 * @param indices Indices of elements which should be assigned the specified value (optional)
 */
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

/*!
 * @brief Kernel function to increase/decrease the elements of the vector (all or only with certain indices) by one specified increment/decrement
 * 
 * @tparam T Data type
 * @param n Number of element to increase/decrease their value
 * @param array Raw pointer to the target array to be modified
 * @param value The increase/decrease for each element's value
 * @param indices Indices of elements for which the value should be modified (optional)
 */
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

/*!
 * @brief Get the amount of free device memory
 * 
 * @return size_t Amount of free device memory in bytes
 * 
 * Message is printed with total and free amounts of device memory in MBytes
 */
size_t requestFreeDeviceMemoryAmount();

#endif // CUDA_HELPER_CUH