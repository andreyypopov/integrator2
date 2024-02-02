#include "cuda_math.cuh"

#include "constants.h"

__host__ __device__ double angle(const Point3 &v1, const Point3 &v2){
    const double den = sqrt(vector_length2(v1) * vector_length2(v2));
    
    if(den < CONSTANTS::EPS_ZERO)
        return 0;

    double res = dot(v1, v2) / den;
    if(res >= 1.0)
        return 0;
    if(res <= -1.0)
        return CONSTANTS::PI;

    return acos(res);
}
