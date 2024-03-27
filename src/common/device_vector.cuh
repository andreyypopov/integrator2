#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#include "cuda_memory.cuh"

template<class T>
struct deviceVector
{
public:
    T *data = nullptr;
    int size = 0;
    int capacity = 0;

    ~deviceVector(){
        free();
    }

    void allocate(int new_size){
        allocate_device(&data, new_size);
        size = new_size;
        capacity = new_size;
    }

    size_t bytes() const {
        return size * sizeof(T);
    }

    void swap(deviceVector<T> &other){
        T *tmpData = other.data;
        int tmpSize = other.size;
        int tmpCapacity = other.capacity;

        other.data = this->data;
        other.size = this->size;
        other.capacity = this->capacity;

        this->data = tmpData;
        this->size = tmpSize;
        this->capacity = tmpCapacity;
    }

    void resize(int new_size){
        if(new_size <= capacity)
            size = new_size;
        else {
            free();

            allocate(CONSTANTS::MEMOTY_REALLOCATION_COEFFICIENT * new_size);
            size = new_size;

            printf("Reallocation performed\n");
        }
    }

private:
    void free(){
        if(data){
            free_device(data);
            data = nullptr;
            size = 0;
            capacity = 0;
        }
    }
};

#endif // DEVICE_VECTOR_CUH
