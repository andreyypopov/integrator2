#ifndef CUDA_MEMORY_CUH
#define CUDA_MEMORY_CUH

#include "cuda_helper.cuh"

template<class T>
void allocate_device(T** ptr, size_t elements_num){
    if(elements_num)
        checkCudaErrors(cudaMalloc(ptr, elements_num * sizeof(T)));
    else
        printf("Zero device memory allocation requested\n");
}

template<class T>
void allocate_host(T** ptr, size_t elements_num){
    if(elements_num)
        checkCudaErrors(cudaMallocHost(ptr, elements_num * sizeof(T)));
    else
        printf("Zero host memory allocation requested\n");
}

template<class T>
void free_device(T* ptr){
    checkCudaErrors(cudaFree(ptr));
}

template<class T>
void free_host(T* ptr){
    checkCudaErrors(cudaFreeHost(ptr));
}

template<class T>
void zero_value_device(T* ptr, size_t elements_num, cudaStream_t stream = nullptr){
    checkCudaErrors(cudaMemsetAsync(ptr, 0, elements_num * sizeof(T), stream));
}

template<class T>
void copy_h2d(const T* src, const T* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, elements_num * sizeof(T), cudaMemcpyHostToDevice, stream));
    else
        printf("Zero host-to-device memory copy requested\n");
}

template<class T>
void copy_d2h(const T* src, const T* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, elements_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
    else
        printf("Zero host-to-device memory copy requested\n");
}

template<class T>
void copy_d2d(const T* src, const T* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, elements_num * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    else
        printf("Zero device-to-device memory copy requested\n");
}

template<class T>
void copy_h2const(const T* src, const void* dst, size_t elements_num, cudaStream_t stream = nullptr){
    if(elements_num)
        checkCudaErrors(cudaMemcpyToSymbolAsync(dst, (void*)src, elements_num * sizeof(T), 0, cudaMemcpyHostToDevice, stream));
    else
        printf("Zero host-to-constant memory copy requested\n");
}

#endif // CUDA_MEMORY_CUH