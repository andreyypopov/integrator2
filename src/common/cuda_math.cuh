#ifndef CUDA_MATH_CUH
#define CUDA_MATH_CUH

typedef double3 Point3;

__host__ __device__ inline Point3 operator+(const Point3 &v1, const Point3 &v2){
    return Point3({ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z });
}

__host__ __device__ inline Point3 operator-(const Point3 &v1, const Point3 &v2){
    return Point3({ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z });
}

__host__ __device__ inline Point3 operator*(const Point3 &v, double a){
    return Point3({ v.x * a, v.y * a, v.z * a });
}

__host__ __device__ inline double dot(const Point3 &v1, const Point3 &v2){
    return v1.x * v2.x + v1.x * v2.y + v1.z * v2.z;
}

__host__ __device__ inline Point3 cross(const Point3 &v1, const Point3 &v2){
    Point3 res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;

    return res;
}

__host__ __device__ inline double vector_length(const Point3 &v){
    return sqrt(dot(v, v));
}

__host__ __device__ inline Point3 normalize(const Point3 &v){
    const double invOldLength = 1.0 / vector_length(v);

    Point3 res;
    res.x = v.x * invOldLength;
    res.y = v.y * invOldLength;
    res.z = v.z * invOldLength;

    return res;
}

#endif // CUDA_MATH_CUH