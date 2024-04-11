#ifndef CONSTANTS_H
#define CONSTANTS_H

/*!
 * @brief Structure with static constant fields for values with different purposes (tolerance, mathematical and other parameters) * 
 */
struct CONSTANTS {
    //tolerance parameters
    static constexpr double DOUBLE_MIN = 2e-6;                              //!< Tolerance for check of double numbers against zero in mathematical functions
    static constexpr double EPS_ZERO = 1e-6;                                //!< Tolerance for check of double numbers against zero in the integration procedure
    static constexpr double EPS_ZERO2 = 1e-10;                              //!< Tolerance squared
    static constexpr double EPS_PSI_THETA = EPS_ZERO;                       //!< Tolerance for check of closeness of angles to \f$\pm\pi\f$
    static constexpr double EPS_PSI_THETA2 = EPS_PSI_THETA * EPS_PSI_THETA; //!< Tolerance squared used in check \f$1+\cos\delta < \frac{\varepsilon^2}2 \f$
    static constexpr double EPS_INTEGRATION = 1e-5;                         //!< Tolerance for check of criterion in the Runge rule

    //mathematical constants
    static constexpr double ONE_THIRD = 0.3333333333333333;                 //!< Math constant \f$\frac13\f$
    static constexpr double PI = 3.14159265358979323846;                    //!< Math constant \f$\pi\f$
    static constexpr double TWO_PI = 6.28318530717958647692;                //!< Math constant \f$2\pi\f$
    static constexpr double RECIPROCAL_FOUR_PI = 0.079577471545947667884;   //!< Math constant \f$\frac1{4\pi}\f$, used in integration

    //algorithmic and program/memory assumptions
    /*!
     * @brief Maximum total number of simple neighbors for a single cell.
     * 
     * Used for memory allocation of a vector of simple neighbors before pairs of them are determined.
     */
    static constexpr int MAX_SIMPLE_NEIGHBORS_PER_CELL = 12;

    /*!
     * @brief Factor for maximum possible number of tasks in case of adaptive mesh control procedure
     * 
     * Used before the start of the integration pipeline to allocate the memory for vectors of integration tasks, 
     * their results and temporary buffers (when the exact number of tasks is not known beforehand).
     */
    static constexpr int MAX_AUTO_REFINEMENT_TASK_COEFFICIENT = 4;

    /*!
     * @brief Coefficient for some additional space which is added to the vector capacity during reallocation.
     * 
     * The constant is not used during original memory allocation for a device vector (exact size is used), but is used
     * in case of further reallocation so as not to perform it too often.
     */
    static constexpr double MEMORY_REALLOCATION_COEFFICIENT = 1.25;

    //limiting constants
    /*!
     * @brief Maximum possible refinement level in case of adaptive mesh control procedure
     * 
     * Used as a termination criterion in the integration cycle and for memory allocation of several device vectors.
     */
    static constexpr int MAX_REFINE_LEVEL = 5;

    /*!
     * @brief Maximum possible number of Gauss points used in a quadrature formula
     * 
     * Used for specification of size of static arrays and constant memory. It is supposed that quadrature formulas
     * with a greater number of Gauss points are not used.
     */
    static constexpr int MAX_GAUSS_POINTS = 13;
};

#endif // CONSTANTS_H
