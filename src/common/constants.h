#ifndef CONSTANTS_H
#define CONSTANTS_H

struct CONSTANTS {
    //tolerance parameters
    static constexpr double DOUBLE_MIN = 2e-6;
    static constexpr double EPS_ZERO = 1e-6;
    static constexpr double EPS_ZERO2 = EPS_ZERO * EPS_ZERO;
    static constexpr double EPS_PSI_THETA = EPS_ZERO;
    static constexpr double EPS_PSI_THETA2 = EPS_PSI_THETA * EPS_PSI_THETA;
    static constexpr double EPS_INTEGRATION = 1e-5;

    //mathematical constants
    static constexpr double ONE_THIRD = 0.3333333333333333;
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double TWO_PI = 6.28318530717958647692;
    static constexpr double RECIPROCAL_FOUR_PI = 0.079577471545947667884;

    //algorithmic and program/memory assumptions
    static constexpr int MAX_SIMPLE_NEIGHBORS_PER_CELL = 12;
    static constexpr int MAX_AUTO_REFINEMENT_TASK_COEFFICIENT = 4;
    static constexpr double MEMOTY_REALLOCATION_COEFFICIENT = 1.25;

    //limiting constants
    static constexpr int MAX_REFINE_LEVEL = 5;
    static constexpr int MAX_GAUSS_POINTS = 13;
};

#endif // CONSTANTS_H
