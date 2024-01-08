#ifndef CONSTANTS_H
#define CONSTANTS_H

struct CONSTANTS {
    static constexpr double DOUBLE_MIN = 2e-6;
    static constexpr double EPS_ZERO = 1e-6;
    static constexpr double EPS_ZERO2 = EPS_ZERO * EPS_ZERO;
    static constexpr double EPS_PSI_THETA = EPS_ZERO;
    static constexpr double EPS_PSI_THETA2 = EPS_PSI_THETA * EPS_PSI_THETA;

    static constexpr double ONE_THIRD = 0.3333333333333333;
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double RECIPROCAL_FOUR_PI = 0.079577471545947667884;

    static constexpr int MAX_SIMPLE_NEIGHBORS_PER_VERTEX = 10;

    static constexpr int MAX_REFINE_LEVEL = 5;
    static constexpr int MAX_GAUSS_POINTS = 13;
};

#endif // CONSTANTS_H