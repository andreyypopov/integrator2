#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#include "cuda_memory.cuh"

template<class T>
struct deviceVector
{
public:
    T *data = nullptr;
    int size = 0;

    ~deviceVector(){
        free();
    }

    void allocate(int new_size){
        allocate_device(&data, new_size);
        size = new_size;
    }

    size_t bytes() const {
        return size * sizeof(T);
    }

    void swap(deviceVector<T> &other){
        T *tmpData = other.data;
        int tmpSize = other.size;

        other.data = this->data;
        other.size = this->size;

        this->data = tmpData;
        this->size = tmpSize;
    }

private:
    void free(){
        if(data){
            free_device(data);
            data = nullptr;
            size = 0;
        }
    }
};

#endif // DEVICE_VECTOR_CUH
