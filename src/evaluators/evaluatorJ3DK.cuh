/*!
 * @file evaluatorJ3DK.cuh
 * @brief EvaluatorJ3DK class for computation of integral of Newtonian potential gradient 
 */
#ifndef EVALUATORJ3DK_CUH
#define EVALUATORJ3DK_CUH

#include "evaluator3d.cuh"

/*!
 * @brief Computation of the functions \f$\mathbf{\Psi}\f$ and \f$\Theta\f$ in the integral \f$\mathbf{J}(M_i,K_j)\f$ of Newtonian potential gradient
 * 
 * @param pt Observation point \f$M_i\f$
 * @param vertices Global vector of vertex coordinates
 * @param triangle Indices of vertices in triangle
 * @return A double4 value with \f$\mathbf{\Psi}\f$ and \f$\Theta\f$
 * 
 * For checks of \f$\cos\varphi^c_a\to1\f$, etc., checks of \f$1 - \cos\varphi^c_a < \frac{\varepsilon^2}2\f$, etc. are performed instead.
 */
__device__ double4 thetaPsi(const Point3 &pt, const Point3 *vertices, const int3 &triangle);

/*!
 * @brief Expressions for singular parts \f$\Theta^{\text{sing}}(M_i,K_j)\f$ and \f${\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)\f$ 
 *  of integral \f$\mathbf{J}_{3D}(K_i,K_j)\f$ for attached neighbors
 * 
 * @param pt Observation point \f$M_i\f$
 * @param i Control panel \f$\bigtriangleup_i\f$
 * @param j Influencing panel \f$\bigtriangleup_j\f$
 * @param vertices Vector of vertex coordinates in the original mesh
 * @param cells Vector of vertex indices in the cells of the original mesh
 * @return Values of functions \f$\Theta^{\text{sing}}(M_i,K_j)\f$ and \f${\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)\f$ at location \f$M_i\f$
 * 
 * The function is used in the kernel function ::kIntegrateRegularPartAttached for integration of the regular part for attached neighbors.
 */
__device__ double4 singularPartAttached(const Point3 &pt, int i, int j, const Point3 *vertices, const int3 *cells);

/*!
 * @brief Expressions for singular parts \f$\Theta^{\text{sing}}(M_i,K_j)\f$ and \f${\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)\f$
 *  of integral \f$\mathbf{J}_{3D}(K_i,K_j)\f$ for simple neighbors
 * 
 * @param pt Observation point \f$M_i\f$
 * @param i Control panel \f$\bigtriangleup_i\f$
 * @param j Influencing panel \f$\bigtriangleup_j\f$
 * @param vertices Vector of vertex coordinates in the original mesh
 * @param cells Vector of vertex indices in the cells of the original mesh
 * @param normals Vector of normals for the cells of the original mesh
 * @param measures Vector of measures for the cells of original mesh
 * @return Values of functions \f$\Theta^{\text{sing}}(M_i,K_j)\f$ and \f${\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)\f$ at location \f$M_i\f$
 * 
 * The function is used in the kernel function ::kIntegrateRegularPartSimple for integration of the regular part for simple neighbors.
 */
__device__ double4 singularPartSimple(const Point3 &pt, int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures);

/*!
 * @brief Computation of integrals of singular part of integrand \f$\int\limits_{K_i} \Theta^{\text{sing}}(M_i,K_j)dS_r\f$ and \f$\int\limits_{K_i} {\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)dS_r\f$
 *      for the attached neighbor pairs
 * 
 * @param i Index of the control panel \f$\bigtriangleup_i\f$
 * @param j Index of the influencing panel \f$\bigtriangleup_j\f$
 * @param vertices Vector of vertex coordinates in the original mesh
 * @param cells Vector of vertex indices in the cells of the original mesh
 * @param normals Vector of cell normals for the original mesh
 * @param measures Vector of cell measures for the original mesh
 * @return Values of \f$\Theta\f$ and \f$\mathbf{\Psi}\f$ parts of the integrals
 * 
 * Scalar part is evaluated as
 * \f[
 *      \int\limits_{K_i} \Theta^{\text{sing}}(M_i,K_j)dS_r = S_i \left(q_{\Theta}(\xi, \alpha, \beta, \gamma, \mu, \lambda) + 
 *              q_{\Theta}(\xi, \beta, \alpha, \delta, \sigma, \theta)\right),
 * \f]
 * while vector part is evaluated as
 * \f[
 *      \int\limits_{K_i} {\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)dS_r = S_i \left(q_{\Psi}(\xi, \alpha, \beta, \gamma, \mu, \lambda)\mathbf{\tau}_b + 
 *              q_{\Psi}(\xi, \beta, \alpha, \delta, \sigma, \theta)\mathbf{\tau}_a - q_{\alpha\beta}(\alpha,\beta)\mathbf{\tau}_c\right).
 * \f]
 * Values of \f$q_{\Theta}\f$ and \f$q_{\Psi}\f$ are calculated by separate functions (::q_thetaPsi and ::q_thetaPsi_zero
 * depending on relative position of panels \f$\bigtriangleup_i, \bigtriangleup_j\f$). Value of \f$q_{\alpha\beta}\f$
 * is calculated directly in this function. \f$S_i\f$ is the area of triangle \f$K_i\f$.
 */
__device__ double4 integrateSingularPartAttached(int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures);

/*!
 * @brief Computation of integrals of singular part of integrand \f$\int\limits_{K_i} \Theta^{\text{sing}}(M_i,K_j)dS_r\f$ and \f$\int\limits_{K_i} {\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)dS_r\f$
 *      for the simple neighbor pairs
 * 
 * @param i Index of the control panel \f$\bigtriangleup_i\f$
 * @param j Index of the influencing panel \f$\bigtriangleup_j\f$
 * @param vertices Vector of vertex coordinates in the original mesh
 * @param cells Vector of vertex indices in the cells of the original mesh
 * @param normals Vector of cell normals for the original mesh
 * @param measures Vector of cell measures for the original mesh
 * @return Values of \f$\Theta\f$ and \f$\mathbf{\Psi}\f$ parts of the integrals
 * 
 * Scalar part is evaluated as
 * \f[
 *      \int\limits_{K_i} \Theta^{\text{sing}}(M_i,K_j)dS_r = S_i \left(q^{\Theta}(\delta_a) - q^{\Theta}(\delta_b) + 4\pi p\right),
 * \f]
 * while vector part is evaluated as
 * \f[
 *      \int\limits_{K_i} {\mathbf{\Psi}}^{\text{sing}}(M_i,K_j)dS_r = S_i \left(q^{\Psi}(\delta_a)\mathbf{\tau}_a + 
 *              q^{\Psi}(\delta_b)\mathbf{\tau}_b\right).
 * \f]
 * Values of \f$q_{\Theta}\f$ and \f$q_{\Psi}\f$ are calculated by a separate function ::q_thetaPsi_cont.
 * The \f$4\pi p\f$ term is added later, when transforming the value of integral from double4 to a Point3
 * (::kFinalizeSimpleNeighborsResults kernel function). \f$S_i\f$ is the area of triangle \f$K_i\f$.
 */
__device__ double4 integrateSingularPartSimple(int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures);

/*!
 * @brief Determine necessary index shifts for pair of simple neighbor triangles (positions of their common vertex in each triangle)
 * 
 * @param triangle1 Indices of the first triangle
 * @param triangle2 Indices of the second triangle
 * @return Shifts for the simple neighbor triangles
 */
__device__ int2 shiftsForSimpleNeighbors(const int3 &triangle1, const int3 &triangle2);

/*!
 * @brief Determine necessary index shifts for pair of attached neighbor triangles (positions of the third vertex, opposite to the common edge, in each triangle)
 * 
 * @param triangle1 Indices of the first triangle
 * @param triangle2 Indices of the second triangle
 * @return Shifts for the attached neighbor triangles
 */
__device__ int2 shiftsForAttachedNeighbors(const int3 &triangle1, const int3 &triangle2);

/*!
 * @brief Generate a new triangle with indices obtained by shifting the indices of an existing one
 * 
 * @param triangle Indices of an existing triangle
 * @param shift Size of shift
 * @return Shifted indices
 */
__device__ int3 rotateLeft(const int3 &triangle, int shift);

/*!
 * @brief Descendent class of Evaluator3D for calculation of repeated integrals of gradient of Newtonian potential \f$\mathbf{J}(K_i,K_j)\f$
 * 
 */
class EvaluatorJ3DK : public Evaluator3D
{
public:
    /*!
     * @brief Construct a new EvaluatorJ3DK object
     * 
     * @param mesh_ Mesh3D object with a loaded surface mesh
     * @param numIntegrator_ NumericalIntegrator3D object with a previously initialized quadrature formula object
     * 
     * The parent class constructor is called without any additional actions
     */
    EvaluatorJ3DK(const Mesh3D &mesh_, NumericalIntegrator3D &numIntegrator_)
        : Evaluator3D(mesh_, numIntegrator_){ };

    /*!
     * @brief Integration over simple neighbor cell pairs
     * 
     * Integration is performed using the following procedure:
     * -# Regular part is integrated numerically. Results are stored in the double4 vector d_simpleNeighborsIntegrals
     * (with separate values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$).
     * -# Singular part is integrated in one sweep for all pairs \f$(i,j)\f$ of original integration tasks using a specific kernel function
     * with analytical formulae. These values are added to the regular part.
     * -# The double4 values are transformed into a single Point3 value for each integral by another kernel function.
     */
    virtual void integrateOverSimpleNeighbors() override;

    /*!
     * @brief Integration over attached neighbor cell pairs
     * 
     * Integration is performed using the following procedure:
     * -# Regular part is integrated numerically. Results are stored in the double4 vector d_attachedNeighborsIntegrals
     * (with separate values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$).
     * -# Singular part is integrated in one sweep for all pairs \f$(i,j)\f$ of original integration tasks using a specific kernel function
     * with analytical formulae. These values are added to the regular part.
     * -# The double4 values are transformed into a single Point3 value for each integral by another kernel function.
     */
    virtual void integrateOverAttachedNeighbors() override;
    
    /*!
     * @brief Integration over non-neighboring cell pairs
     * 
     * Only numerical integration is performed as integrals are non-singular. Then the double4 d_notNeighborsIntegrals values
     * (with separate values for \f$\mathbf{\Psi}\f$ and \f$\Theta\f$) are transformed into a single Point3 value for each integral
     * by a separate kernel function.
     */
    virtual void integrateOverNotNeighbors() override;

private:
    /*!
     * @brief Numerical integration (for all types of neighbor cells)
     * 
     * @param neighborType Type of neighbors
     * 
     * Regardless of the error control mode one iteration of numerical integration is performed and the results are summed up for the refined cells.
     * Numerical integration is implemented as a Lambda function which calls a kernel for a specific neighbor type. All 3 kernel function
     * differ only in the integrand function (procedure of application of Gaussian quadrature is the same).
     * 
     * If adaptive error control procedure is used, the following loop is run:
     * -# Current mesh cells are refined (only those cells which need to be refinement, whole mesh at iteration 1).
     * -# Numerical integration is performed over new tasks.
     * -# Results of integration over refined cells are summed up for the original tasks.
     * -# Results of integration of 2 last iterations are compared using Runge rule, a list of remaining tasks is filled.
     * -# If there tasks left for further integration, a list of cells for refinement is filled.
     * 
     * The loop stops when no tasks are left which have not converged or the maximum number of itrations has been reached.
     * 
     * Additionally, in case of adaptive error control the mesh is reset to the original state before the first iteration
     * of integration as it may be completely different after previous integration for another type of neighbors
     * (may contain refined cells, which may also cover only part of the surface depending on the integral convergence).
     */
    void numericalIntegration(neighbour_type_enum neighborType);
};

#endif // EVALUATORJ3DK_CUH
