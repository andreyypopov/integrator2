#include "cuda_math.cuh"

#include "constants.h"

/*!
 * If product of vector lengths is less than EPS_ZERO constant then 0 is returned. Otherwise angle is computed as
 * \f$ \arccos\frac{\mathbf a\cdot\mathbf b}{|\mathbf a|\cdot|\mathbf b|} \f$. If for the reason of numerical errors
 * the arccosine argument is greater than 1.0 then zero angle is returned, if the argument is less than -1.0 then \f$\pi\f$ is returned.
 */
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
