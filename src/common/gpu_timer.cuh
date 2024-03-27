#ifndef GPU_TIMER_CUH
#define GPU_TIMER_CUH

#include "cuda_helper.cuh"

class GpuTimer
{
public:
    GpuTimer(){
        checkCudaErrors(cudaEventCreate(&startEvent));
        checkCudaErrors(cudaEventCreate(&stopEvent));
    }

    ~GpuTimer(){
        if(startEvent)
            checkCudaErrors(cudaEventDestroy(startEvent));
        if(stopEvent)
            checkCudaErrors(cudaEventDestroy(stopEvent));
    }

    void start(cudaStream_t stream = nullptr){
        if(startEvent)
            checkCudaErrors(cudaEventRecord(startEvent, stream));
    }

    float stop(const char *message = nullptr, cudaStream_t stream = nullptr){
        if(stopEvent){
            checkCudaErrors(cudaEventRecord(stopEvent, stream));
            checkCudaErrors(cudaEventSynchronize(stopEvent));

            float time;
            checkCudaErrors(cudaEventElapsedTime(&time, startEvent, stopEvent));

            if(message)
                printf("Time for %s: %6.3f ms\n", message, time);

            return time;
        } else
            return 0;
    }

private:
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
};

#endif // GPU_TIMER_CUH
