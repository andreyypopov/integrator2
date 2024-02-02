#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#include "cuda_memory.cuh"

template<class T>
struct deviceVector
{
    T *data = nullptr;
    int size = 0;

    void allocate(int new_size){
        allocate_device(&data, new_size);
        size = new_size;
    }

    void free(){
        if(data)
            free_device(data);
    }

    size_t bytes() const {
        return size * sizeof(T);
    }
};

#endif // DEVICE_VECTOR_CUH
