#ifndef CUDA_MATH_CUH
#define CUDA_MATH_CUH

#include "constants.h"

/*!
 * @brief Typedef for 3D vector
 */
typedef double3 Point3;

/*!
 * @brief Square of a real number
 * 
 * @param x Scalar value
 * @return Value of the input scalar
 */
__host__ __device__ inline double sqr(double x){
    return x * x;
}

/*!
 * @brief Sign of a real number
 * 
 * @param x Scalar value
 * @return Sign of the input scalar
 */
__host__ __device__ inline double sign(double x){
    if (fabs(x) < CONSTANTS::DOUBLE_MIN)
        return 0.0;
        
    return (x > CONSTANTS::DOUBLE_MIN) ? 1.0 : -1.0;
}

/*!
 * @brief Calculate \f$\arg x\f$
 * 
 * @param x Argument of the function
 * @return Value of the \f$\arg\f$ function
 * 
 * Formula \f$\arg x = \mathrm{arctg}(0,x)\f$ is used for calculation, which yields either \f$0\f$, if \f$x>0\f$
 * or \f$\pi\f$, if \f$x<0\f$.
 */
__host__ __device__ inline double arg(double x){
    return (x > CONSTANTS::DOUBLE_MIN) ? 0.0 : CONSTANTS::PI;
}

/*!
 * @brief Construct a double4 with prescribed vector part
 * 
 * @param v Vector part
 * @return Resulting double4 value
 */
__host__ __device__ inline double4 assign_vector_part(const Point3 &v){
    return make_double4(v.x, v.y, v.z, 0);
}

/*!
 * @brief Extract vector part from a double4 value
 * 
 * @param v double4 value
 * @return Vector part of the input value
 */
__host__ __device__ inline Point3 extract_vector_part(const double4 &v){
    return make_double3(v.x, v.y, v.z);
}

/*!
 * @brief Sum of 2 double4 values
 * 
 * @param v1 First double4 operand
 * @param v2 Second double4 operand
 * @return Resulting sum
 */
__host__ __device__ inline double4 operator+(const double4 &v1, const double4 &v2){
    return double4({ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w });
}

/*!
 * @brief Subtraction of one double4 value from another one
 * 
 * @param v1 First double4 operand
 * @param v2 Second double4 operand
 * @return Resulting double4 value
 */
__host__ __device__ inline double4 operator-(const double4 &v1, const double4 &v2){
    return double4({ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w });
}

/*!
 * @brief Multiplication of a double4 value by a scalar
 * 
 * @param a Scalar value
 * @param v double4 value
 * @return Resulting double4 value
 */
__host__ __device__ inline double4 operator*(double a, const double4 &v){
    return double4({ v.x * a, v.y * a, v.z * a, v.w * a });
}

/*!
 * @brief Add a double4 value to a current one
 * 
 * @param v Current double4 value
 * @param a double4 value to be added
 */
__host__ __device__ inline void operator+=(double4 &v, const double4 &a){
    v.x += a.x;
    v.y += a.y;
    v.z += a.z;
    v.w += a.w;
}

/*!
 * @brief Sum of 2 Point3 values
 * 
 * @param v1 First Point3 operand
 * @param v2 Second Point3 operand
 * @return Resulting sum
 */
__host__ __device__ inline Point3 operator+(const Point3 &v1, const Point3 &v2){
    return Point3({ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z });
}

/*!
 * @brief Subtraction of one Point3 value from another one
 * 
 * @param v1 First Point3 operand
 * @param v2 Second Point3 operand
 * @return Resulting Point3 value
 */
__host__ __device__ inline Point3 operator-(const Point3 &v1, const Point3 &v2){
    return Point3({ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z });
}

/*!
 * @brief Negation of a Point3 value
 * 
 * @param v Input Point3 value
 * @return Negated value
 */
__host__ __device__ inline Point3 operator-(const Point3 &v){
    return Point3({ -v.x, -v.y, -v.z });
}

/*!
 * @brief Multiplication of a Point3 value by a scalar
 * 
 * @param a Scalar value
 * @param v Point3 value
 * @return Resulting Point3 value
 */
__host__ __device__ inline Point3 operator*(double a, const Point3 &v){
    return Point3({ v.x * a, v.y * a, v.z * a });
}

/*!
 * @brief Dot product of 2 3D vectors
 * 
 * @param v1 First 3D vector
 * @param v2 Second 3D vector
 * @return Resulting dot product
 */
__host__ __device__ inline double dot(const Point3 &v1, const Point3 &v2){
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

/*!
 * @brief Cross product of 2 3D vectors
 * 
 * @param v1 First 3D vector
 * @param v2 Second 3D vector
 * @return Resulting cross product 
 */
__host__ __device__ inline Point3 cross(const Point3 &v1, const Point3 &v2){
    Point3 res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;

    return res;
}

/*!
 * @brief Length of a 3D vector
 * 
 * @param v Input 3D vector
 * @return Resulting length
 */
__host__ __device__ inline double vector_length(const Point3 &v){
    return sqrt(dot(v, v));
}

/*!
 * @brief Squared length of a 3D vector
 * 
 * @param v Input 3D vector
 * @return Resulting squared length
 */
__host__ __device__ inline double vector_length2(const Point3 &v){
    return dot(v, v);
}

/*!
 * @brief Add a Point3 value to a current one
 * 
 * @param v Current Point3 value
 * @param a Point3 value to be added
 */
__host__ __device__ inline void operator+=(Point3 &v, const Point3 &a){
    v.x += a.x;
    v.y += a.y;
    v.z += a.z;
}

/*!
 * @brief Multiply a Point3 value by a scalar
 * 
 * @param v Current Point3 value
 * @param a Scalar value to multiply by
 */
__host__ __device__ inline void operator*=(Point3 &v, const double &a){
    v.x *= a;
    v.y *= a;
    v.z *= a;
}

/*!
 * @brief Divide a Point3 value by a scalar
 * 
 * @param v Current Point3 value
 * @param a Scalar value to divide by
 */
__host__ __device__ inline void operator/=(Point3 &v, const double &a){
    v.x /= a;
    v.y /= a;
    v.z /= a;
}

/*!
 * @brief Normalize a 3D vector
 * 
 * @param v Input 3D vector
 * @return Resulting unit vector in the direction of the input vector
 */
__host__ __device__ inline Point3 normalize(const Point3 &v){
    const double invOldLength = 1.0 / vector_length(v);

    Point3 res;
    res.x = v.x * invOldLength;
    res.y = v.y * invOldLength;
    res.z = v.z * invOldLength;

    return res;
}

/*!
 * @brief Component-wise division of 2 double4 operands
 * 
 * @param v1 Numerator
 * @param v2 Denominator
 * @return Resulting quotient
 * 
 * In case both numerator and denominator are less than DOUBLE_MIN constant (for vector part components)
 * or less than EPS_ZERO2 constant (for scalar part) the corresponding component is set to zero, other division is performed.
 */
__host__ __device__ inline double4 divide(const double4 &v1, const double4 &v2){
    double4 res;

    res.x = (fabs(v1.x) < CONSTANTS::DOUBLE_MIN && fabs(v2.x) < CONSTANTS::DOUBLE_MIN) ? 0.0 : (v1.x / v2.x);
    res.y = (fabs(v1.y) < CONSTANTS::DOUBLE_MIN && fabs(v2.y) < CONSTANTS::DOUBLE_MIN) ? 0.0 : (v1.y / v2.y);
    res.z = (fabs(v1.z) < CONSTANTS::DOUBLE_MIN && fabs(v2.z) < CONSTANTS::DOUBLE_MIN) ? 0.0 : (v1.z / v2.z);

    res.w = (fabs(v1.w) < CONSTANTS::EPS_ZERO2 && fabs(v2.w) < CONSTANTS::EPS_ZERO2) ? 0.0 : (v1.w / v2.w);

    return res;
}

/*!
 * @brief Compute 1-norm for a 3D vector
 * 
 * @param v Input 3D vector
 * @return Resulting norm
 */
__host__ __device__ inline double norm1(const Point3 &v){
    return fabs(v.x) + fabs(v.y) + fabs(v.z);
}

/*!
 * @brief Compute 1-norm of a double4 vector
 * 
 * @param v Input double4 vector
 * @return Resulting norm
 */
__host__ __device__ inline double norm1(const double4 &v){
    return fabs(v.x) + fabs(v.y) + fabs(v.z) + fabs(v.w);
}

/*!
 * @brief Compute angle between 2 3D vectors
 * 
 * @param v1 First 3D vector
 * @param v2 Second 3D vector
 * @return Resulting angle between vectors
 */
__host__ __device__ double angle(const Point3 &v1, const Point3 &v2);

#endif // CUDA_MATH_CUH