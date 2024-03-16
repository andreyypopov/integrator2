#include "evaluatorJ3DK.cuh"

__global__ void kIntegrateSingularPartAttached(int n, double4 *results, const int3 *tasks, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = tasks[idx];

        const double4 singularThetaPsi = integrateSingularPartAttached(task.x, task.y, vertices, cells, normals, measures);
        results[idx] += singularThetaPsi;
    }
}

__global__ void kIntegrateSingularPartSimple(int n, double4 *results, const int3 *tasks, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = tasks[idx];

        const double4 singularThetaPsi = integrateSingularPartSimple(task.x, task.y, vertices, cells, normals, measures);
        results[idx] += singularThetaPsi;
    }
}

__global__ void kIntegrateRegularPartSimple(int n, double4 *integrals, const int3 *refinedTasks, const int3 *originalTasks,
            const Point3 *vertices, const int3 *cells, const Point3 *cellNormals, const double *cellMeasures,
            const Point3 *refinedVertices, const int3 *refinedCells, const double *refinedCellMeasures, int GaussPointsNum)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = refinedTasks[idx];
        const int3 originalTask = originalTasks[task.z];
        const int3 triangleI = refinedCells[task.x];
        const int3 triangleJ = cells[task.y];

        Point3 quadraturePoints[CONSTANTS::MAX_GAUSS_POINTS];
        double4 functionValues[CONSTANTS::MAX_GAUSS_POINTS];

        calculateQuadraturePoints(quadraturePoints, refinedVertices, triangleI);
        for(int i = 0; i < GaussPointsNum; ++i)
            functionValues[i] = thetaPsi(quadraturePoints[i], vertices, triangleJ) - singularPartSimple(quadraturePoints[i], originalTask.x, task.y, vertices, cells, cellNormals, cellMeasures);

        const double4 res = integrate4D(functionValues);
        integrals[idx] = refinedCellMeasures[task.x] * res;
    }
}

__global__ void kIntegrateRegularPartAttached(int n, double4 *integrals, const int3 *refinedTasks, const int3 *originalTasks,
            const Point3 *vertices, const int3 *cells, const Point3 *cellNormals,
            const Point3 *refinedVertices, const int3 *refinedCells, const double *refinedCellMeasures, int GaussPointsNum)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = refinedTasks[idx];
        const int3 originalTask = originalTasks[task.z];
        const int3 triangleI = refinedCells[task.x];
        const int3 triangleJ = cells[task.y];

        Point3 quadraturePoints[CONSTANTS::MAX_GAUSS_POINTS];
        double4 functionValues[CONSTANTS::MAX_GAUSS_POINTS];

        calculateQuadraturePoints(quadraturePoints, refinedVertices, triangleI);
        for(int i = 0; i < GaussPointsNum; ++i)
            functionValues[i] = thetaPsi(quadraturePoints[i], vertices, triangleJ) - singularPartAttached(quadraturePoints[i], originalTask.x, task.y, vertices, cells);

        const double4 res = integrate4D(functionValues);
        integrals[idx] = refinedCellMeasures[task.x] * res;
    }
}

__global__ void kIntegrateNotNeighbors(int n, double4 *integrals, const int3 *refinedTasks, const Point3 *vertices, const int3 *cells,
            const Point3 *refinedVertices, const int3 *refinedCells, const double *refinedCellMeasures, int GaussPointsNum)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = refinedTasks[idx];
        const int3 triangleI = refinedCells[task.x];
        const int3 triangleJ = cells[task.y];

        Point3 quadraturePoints[CONSTANTS::MAX_GAUSS_POINTS];
        double4 functionValues[CONSTANTS::MAX_GAUSS_POINTS];

        calculateQuadraturePoints(quadraturePoints, refinedVertices, triangleI);
        for(int i = 0; i < GaussPointsNum; ++i)
            functionValues[i] = thetaPsi(quadraturePoints[i], vertices, triangleJ);

        const double4 res = integrate4D(functionValues);
        integrals[idx] = refinedCellMeasures[task.x] * res;
    }
}

__global__ void kFinalizeSimpleNeighborsResults(int n, Point3 *results, const double4 *integrals, const int3 *tasks, const Point3 *cellNormals, const double *cellMeasures)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = tasks[idx];
        const double4 integral = integrals[idx];
        const Point3 normal = cellNormals[task.y];

        int p = 0;
        const double refTheta = CONSTANTS::TWO_PI * cellMeasures[task.y];
        if(integral.w > refTheta)
            p = -((int)trunc((integral.w - refTheta) / (2.0 * refTheta)) + 1);
        else if(integral.w < -refTheta)
            p = ((int)trunc((-refTheta - integral.w) / (2.0 * refTheta)) + 1);

        results[idx] = CONSTANTS::RECIPROCAL_FOUR_PI * ((integral.w + 2.0 * p * refTheta) * normal + cross(extract_vector_part(integral), normal));
    }
}

__global__ void kFinalizeNonSimpleResults(int n, Point3 *results, const double4 *integrals, const int3 *tasks, const Point3 *cellNormals)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        const int3 task = tasks[idx];
        const double4 integral = integrals[idx];
        const Point3 normal = cellNormals[task.y];

        results[idx] = CONSTANTS::RECIPROCAL_FOUR_PI * (integral.w * normal + cross(extract_vector_part(integral), normal));
    }
}

__device__ double4 thetaPsi(const Point3 &pt, const Point3 *vertices, const int3 &triangle)
{
    double4 res;

    const Point3 triA = vertices[triangle.x];
    const Point3 triB = vertices[triangle.y];
    const Point3 triC = vertices[triangle.z];

    Point3 ova = pt - triA;
    Point3 ovb = pt - triB;
    Point3 ovc = pt - triC;

    double lva = vector_length(ova);
    double lvb = vector_length(ovb);
    double lvc = vector_length(ovc);

    ova /= lva;
    ovb /= lvb;
    ovc /= lvc;

    const Point3 taua = normalize(triC - triB);
    const Point3 taub = normalize(triA - triC);
    const Point3 tauc = normalize(triB - triA);

    const double rac = dot(ova, tauc), rbc = dot(ovb, tauc), rba = dot(ovb, taua), rca = dot(ovc, taua), rcb = dot(ovc, taub), rab = dot(ova, taub);

    double term1, term2, term3;

    if(fabs(rbc + 1.0) < 0.5 * CONSTANTS::EPS_PSI_THETA2)
        term1 = log(lvb / lva);
    else
        term1 = log((lva * (1.0 + rac)) / (lvb * (1.0 + rbc)));

    if(fabs(rca + 1.0) < 0.5 * CONSTANTS::EPS_PSI_THETA2)
        term2 = log(lvc / lvb);
    else
        term2 = log((lvb * (1.0 + rba)) / (lvc * (1.0 + rca)));

    if(fabs(rab + 1.0) < 0.5 * CONSTANTS::EPS_PSI_THETA2)
        term3 = log(lva / lvc);
    else
        term3 = log((lvc * (1.0 + rcb)) / (lva * (1.0 + rab)));

    res = assign_vector_part(term1 * tauc + term2 * taua + term3 * taub);
    res.w = 2.0 * atan2(dot(cross(ova, ovb), ovc), 1.0 + dot(ova, ovb) + dot(ovb, ovc) + dot(ovc, ova));

    return res;
}

__device__ double4 singularPartAttached(const Point3 &pt, int i, int j, const Point3 *vertices, const int3 *cells)
{
    double4 res;

    const int3 triangleI = cells[i];
    const int3 triangleJ = cells[j];

    const int2 shifts = shiftsForAttachedNeighbors(triangleI, triangleJ);
    const int3 shiftedTriangleJ = rotateLeft(triangleJ, shifts.y);

    const Point3 triJA = vertices[shiftedTriangleJ.x];
    const Point3 triJB = vertices[shiftedTriangleJ.y];
    const Point3 triJC = vertices[shiftedTriangleJ.z];

    Point3 va = pt - triJB;
    Point3 vb = pt - triJC;

    const double lva = vector_length(va);
    const double lvb = vector_length(vb);

    va /= lva;
    vb /= lvb;

    const Point3 taua = normalize(triJA - triJC);
    const Point3 taub = normalize(triJB - triJA);
    Point3 tauc = triJC - triJB;

    const double ilvc = 1.0 / vector_length(tauc);
    tauc *= ilvc;
    
	res = assign_vector_part(log((lvb * dot(tauc, tauc - vb)) / (lva * dot(tauc, tauc - va))) * tauc -
			 log(lva * dot(taub, taub + va) * ilvc) * taub - log(lvb * dot(taua, taua - vb) * ilvc) * taua);
    res.w = 2.0 * (atan2(dot(cross(va, taub), tauc), dot(taub - tauc, taub + va)) - atan2(dot(cross(vb, taua), tauc), dot(taua - tauc, taua - vb)));

    return res;
}

__device__ double4 singularPartSimple(const Point3 &pt, int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures)
{
    double4 res;

    const int3 triangleI = cells[i];
    const int3 triangleJ = cells[j];

    const int2 shifts = shiftsForSimpleNeighbors(triangleI, triangleJ);
    const int3 shiftedTriangleJ = rotateLeft(triangleJ, shifts.y);

    const Point3 triJA = vertices[shiftedTriangleJ.x];
    const Point3 triJB = vertices[shiftedTriangleJ.y];
    const Point3 triJC = vertices[shiftedTriangleJ.z];

	Point3 ovc = pt - triJA;
	const double lvc = vector_length(ovc);
    ovc /= lvc;

    const Point3 taua = normalize(triJA - triJC);
    const Point3 taub = normalize(triJB - triJA);

    const Point3 normalI = normals[i];
    const Point3 normalJ = normals[j];

    Point3 e = cross(normalI, normalJ);
    if(vector_length2(e) < CONSTANTS::EPS_ZERO2)
		e = taub;
	else
		e = normalize(e);

    auto getDeltas = [taua, taub, normalJ](const Point3 &e){
        double2 res;
        res.x = atan2(dot(cross(taua, e), normalJ), -dot(e, taua));
        res.y = atan2(dot(cross(e, taub), normalJ), dot(e, taub));
        return res;
    };

    double2 delta = getDeltas(e);

    if((CONSTANTS::PI - fabs(delta.x) < CONSTANTS::EPS_ZERO) || (CONSTANTS::PI - fabs(delta.y) < CONSTANTS::EPS_ZERO)){
        e *= -1;
        delta = getDeltas(e);
    }

    if((delta.x * delta.y < 0) && (fabs(delta.x - delta.y) > CONSTANTS::PI)){
        e *= -1;
        delta = getDeltas(e);
    }

	const double invAri = rsqrt(measures[i]);

    res = assign_vector_part(-(log((lvc * (1 + dot(taua,  ovc))) * invAri) * taua + log((lvc * (1 - dot(taub, ovc))) * invAri) * taub));
    res.w = 2.0 * (atan2(dot(cross(ovc, taua), e), dot(e - ovc, e - taua)) + atan2(dot(cross(ovc, taub), e), dot(e - ovc, e + taub)));

	return res;
}

__device__ double phi(double2 sinCosAlpha, double2 sinCosGamma, double sinXi, double cosLambda)
{
    return 2.0 * atan2(sinXi * sinCosAlpha.x * sinCosGamma.x, 1.0 - sinCosAlpha.y + sinCosGamma.y + cosLambda);
}

__device__ double2 q_thetaPsi(double2 sinCosAlpha, double2 sinCosBeta, double2 sinCosGamma, double2 sinCosNu, double2 sinCosXi, double cosMu, double cosLambda)
{
    double2 res;

    const double phi1 = phi(sinCosAlpha,  sinCosGamma, sinCosXi.x, cosLambda);
    const double phi2 = phi(make_double2(sinCosAlpha.x, -sinCosAlpha.y), make_double2(sinCosGamma.x, -sinCosGamma.y), sinCosXi.x, cosLambda);

    const double iSinAlphaSin2mu = 1.0 / (sinCosAlpha.x * (1.0 - cosMu * cosMu));

    res.x = phi1 + sinCosGamma.x * sinCosNu.x * iSinAlphaSin2mu * (
                    (sinCosBeta.y * sinCosGamma.x - sinCosXi.y * sinCosBeta.x * sinCosGamma.y) * phi2 +
                    sinCosXi.x * sinCosBeta.x * (0.5 * (1.0 + cosMu) * log((1.0 + sinCosBeta.y) / (1.0 - sinCosNu.y)) +
					0.5 * (1.0 - cosMu) * log((1.0 - sinCosBeta.y) / (1.0 + sinCosNu.y)) +
					log((1.0 + cosLambda) / (1.0 - sinCosGamma.y))));

    res.y = 1.5 - iSinAlphaSin2mu * (
				sinCosBeta.x * (sinCosNu.y + cosMu * cosLambda) * log(1.0 + cosLambda) +
				sinCosNu.x * (sinCosBeta.y + cosMu * sinCosGamma.y) * log(1.0 - sinCosGamma.y) +
				sinCosBeta.x * (1.0 - cosMu) * (sinCosNu.y - cosLambda) * log(sinCosBeta.x / sinCosNu.x) +
				sinCosNu.x * sinCosBeta.x * (sinCosBeta.x * sinCosGamma.y - sinCosXi.y * sinCosGamma.x * sinCosBeta.y) * log((1.0 - sinCosNu.y) / (1.0 + sinCosBeta.y)) +
				phi2 * sinCosXi.x * sinCosGamma.x * sinCosNu.x * sinCosBeta.x) ;

    return res;
}

__device__ double2 q_thetaPsi_zero(double2 sinCosBeta, double2 sinCosNu, double sinAlpha)
{
    double2 res;
    res.y = 1.5 - (sinCosNu.y * sinCosBeta.x * log(1.0 + sinCosNu.y) + sinCosNu.x * sinCosBeta.y * log(1.0 - sinCosBeta.y) + sinAlpha
                            - sinCosBeta.x + sinCosNu.x + sinCosBeta.x * sinCosNu.y * log(sinCosBeta.x / sinCosNu.x)) / sinAlpha;

    return res;
}

__device__ double2 q_thetaPsi_cont(double2 sinCosXi, double mu, double2 sinCosMu, double logSinMu, double logSinNu,
        double psi, double2 sinCosPsi, double nu, double2 sinCosNu, double kappa, double sinKappa, double sinNuPsi, double sinMuPsi,
        double delta, double2 sinCosDelta, double cosLambda, double cosTheta, double cosEta, double cosSigma, double cosChi)
{
    double2 res;

    const double logOneCosTheta = log(1.0 + cosTheta);
    const double logOneCosLambda = log(1.0 + cosLambda);

    const double Lambda1 = logOneCosLambda - logOneCosTheta + logSinNu - logSinMu;
    const double Lambda2 = log(tan(0.5 * nu) * tan(0.5 * mu));

    double2 sinCos05Delta;
    sincos(0.5 * delta, &sinCos05Delta.x, &sinCos05Delta.y);
    const double tan05Delta = sinCos05Delta.x / sinCos05Delta.y;

    double2 sinCos05MuPsi, sinCos05NuPsi;
    sincos(0.5 * (mu - psi), &sinCos05MuPsi.x, &sinCos05MuPsi.y);
    sincos(0.5 * (nu + psi), &sinCos05NuPsi.x, &sinCos05NuPsi.y);
    const double Amu = atan2(tan05Delta * sinCos05MuPsi.y * sinCosXi.x, tan05Delta * sinCos05MuPsi.y * sinCosXi.y + sinCos05MuPsi.x);
    const double Anu = atan2(tan05Delta * sinCos05NuPsi.x * sinCosXi.x, tan05Delta * sinCos05NuPsi.x * sinCosXi.y + sinCos05NuPsi.y);

    double2 sinCosTmp, sinCosTmp2;
    sincos(0.5 * (mu-psi) - 0.5 * (nu+psi), &sinCosTmp.x, &sinCosTmp.y);
    sincos(0.5 * kappa, &sinCosTmp2.x, &sinCosTmp2.y);

    const double W = atan2(sinCosDelta.x * sinCosTmp2.x * sinCosXi.x, 
			sinCosTmp2.y + sinCosDelta.x * sinCosTmp.y * sinCosXi.y + sinCosDelta.y * sinCosTmp.x);
	
    sincos(delta - psi, &sinCosTmp.x, &sinCosTmp.y);
	const double D = 1.0 / (sqr(sinCosTmp.x) + sinCosDelta.x * sinCosPsi.x * (1.0 - sinCosXi.y) * (sinCosTmp.y + cosSigma));
    const double G = sinCosPsi.y * (sinCosDelta.x * cosSigma * sinCosXi.y + sinCosDelta.y * cosChi - sqr(sinCosDelta.x) / sinCosPsi.x);

	const double gent = 2.0 * (Anu * sinCosMu.x * sinNuPsi - Amu * sinCosNu.x * sinMuPsi -
			D * sinCosMu.x * sinCosNu.x * sinCosDelta.x * (W * cosEta + 0.5 * sinCosPsi.x * sinCosXi.x * (Lambda1 - Lambda2 * cosSigma))) / (sinCosPsi.x * sinKappa);

	const double gens = 0.5 * (3.0 - log(2.0)) +
			(sinCosMu.x * sinCosNu.x / sinKappa) * ((logOneCosLambda - logOneCosTheta) * sinCosPsi.y / sinCosPsi.x + D*(Lambda1 * sinCosDelta.x * cosEta / sinCosPsi.x +
				Lambda2 * cosChi - 2.0 * W * sinCosDelta.x * sinCosXi.x - G * (logSinNu - logSinMu))) -
			(sinCosMu.y * sinCosNu.x * (logOneCosLambda-logSinMu) + sinCosMu.x * sinCosNu.y * (logOneCosTheta-logSinNu))/sinKappa -
			0.5*(logSinMu+logSinNu-log(sinKappa));

	const double mulPsi   = sign(sinCosXi.x * sinCosPsi.x);
	const double mulDelta = sign(sinCosXi.x * sinCosDelta.x);

    if ((fabs(sinCosXi.x) < CONSTANTS::EPS_ZERO) && (1.0 - fabs(cosSigma) < 0.5 * CONSTANTS::EPS_ZERO2) && (fabs(sinCosPsi.x) > CONSTANTS::EPS_ZERO)){
        double ara = arg(sin(0.5 * (nu + psi)) * cosSigma);
        double arb = arg(cos(0.5 * (mu - psi)) * cosSigma);

        res.x = 2.0 * mulPsi * cosSigma * sinCosXi.y / (sinKappa * sinCosPsi.x) * (sinCosMu.x * sinNuPsi * ara - sinCosNu.x * sinMuPsi * arb);
        res.y = 0.5 * (1.0 - log(2.0)) - 0.5 * log((1.0 - cosSigma * sinCosMu.y) * (1.0 + cosSigma * sinCosNu.y) / sinKappa) +
                cosSigma * (sinCosMu.x - sinCosNu.x + 0.5 * sin(mu - nu) * Lambda2) / sinKappa;
        
        return res;
    }

    if ((fabs(sinCosXi.x) < CONSTANTS::EPS_ZERO) && (fabs(sinCosPsi.x) > CONSTANTS::EPS_ZERO)){			
        double are = arg(sin(0.5 * (nu + mu)) + sin(0.5 * (mu - nu) - psi + delta * sinCosXi.y));
        double arc = arg(1.0 / tan(0.5 * (nu + psi)) + tan(0.5 * delta) * sinCosXi.y);
        double ard = arg(      tan(0.5 * (mu - psi)) + tan(0.5 * delta) * sinCosXi.y);		

        res.x = 2.0 * mulDelta * (sinCosDelta.x * sinCosMu.x * sinCosNu.x / cosChi * sinCosXi.y * are +
                                                    sinCosMu.x * sinNuPsi * arc - sinCosNu.x * sinMuPsi * ard ) / (sinKappa * sinCosPsi.x);
        res.y = gens;

        return res;
    }

    if (fabs(sin(psi)) < CONSTANTS::EPS_ZERO){
        if (fabs(delta) > CONSTANTS::EPS_ZERO){
            res.x = 2.0 * (sinCosMu.x * sinCosNu.x * (W * sinCosDelta.y * sinCosXi.y - 0.5 * (Lambda1 - Lambda2 * cosSigma) * sinCosXi.x) / sinCosDelta.x +
                    (Anu + mulDelta*arg(sin(0.5*(nu+psi)))) * sinCosMu.x * sinCosNu.y +
                        (Amu + mulDelta * arg(cos(0.5 * (mu - psi)))) * sinCosMu.y * sinCosNu.x) / sinKappa;
            res.y = 0.5 * (3.0 - log(2.0)) - 0.5 * (log((1.0 + cosLambda) * (1.0 + cosTheta) / sinKappa) - sin(mu - nu)  / sinKappa * Lambda1) -
                    sinCosMu.x * sinCosNu.x * ((Lambda1 * sinCosDelta.y - Lambda2 * sinCosPsi.y) * sinCosXi.y + 2.0 * W * sinCosXi.x) / (sinKappa * sinCosDelta.x);

            return res;
        } else {
            res.x = 2.0 * arg(sinCosPsi.y);
            res.y = 0.5 * (1.0 - log(2.0)) - 0.5 * log((1.0 - sinCosPsi.y * sinCosMu.y) * (1.0 + sinCosPsi.y * sinCosNu.y) / sinKappa) + 
                            (0.5 * sin(mu - nu) * Lambda1 + sinCosPsi.y * (sinCosMu.x - sinCosNu.x)) / sinKappa;

            return res;
        }				
    }

    res.x = gent;
    res.y = gens;

    return res;
}

__device__ double4 integrateSingularPartAttached(int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures)
{
    double4 res;

    const int3 triangleI = cells[i];
    const int3 triangleJ = cells[j];

    const int2 shifts = shiftsForAttachedNeighbors(triangleI, triangleJ);
    const int3 shiftedTriangleI = rotateLeft(triangleI, shifts.x);
    const int3 shiftedTriangleJ = rotateLeft(triangleJ, shifts.y);

    const Point3 triIA = vertices[shiftedTriangleI.x];
    const Point3 triIB = vertices[shiftedTriangleI.y];
    const Point3 triIC = vertices[shiftedTriangleI.z];
    const Point3 triJA = vertices[shiftedTriangleJ.x];
    const Point3 triJB = vertices[shiftedTriangleJ.y];
    const Point3 triJC = vertices[shiftedTriangleJ.z];

    const Point3 taua = normalize(triJA - triJC);
    const Point3 taub = normalize(triJB - triJA);
    const Point3 tauc = normalize(triJC - triJB);

    const double alpha = angle(triIA - triIC, triIB - triIC);
    const double beta  = angle(triIC - triIB, triIA - triIB);
    const double gamma = angle(triJC - triJB, triJA - triJB);
    const double delta = angle(triJB - triJC, triJA - triJC);

    const double nu = CONSTANTS::PI - alpha - beta;

    const Point3 normalI = normals[i];
    const Point3 normalJ = normals[j];

    const double xi = atan2(dot(cross(normalI, normalJ), tauc), dot(normalI, normalJ));
    
    //.x is the sine value, .y is the cosine value
    double2 sinCosAlpha, sinCosBeta, sinCosGamma, sinCosDelta, sinCosXi, sinCosNu;
    sincos(alpha, &sinCosAlpha.x, &sinCosAlpha.y);
    sincos(beta,  &sinCosBeta.x,  &sinCosBeta.y);
    sincos(gamma, &sinCosGamma.x, &sinCosGamma.y);
    sincos(delta, &sinCosDelta.x, &sinCosDelta.y);
    sincos(xi, &sinCosXi.x, &sinCosXi.y);

    sinCosNu.x = sinCosAlpha.x * sinCosBeta.y + sinCosAlpha.y * sinCosBeta.x;
    sinCosNu.y = sinCosAlpha.x * sinCosBeta.x - sinCosAlpha.y * sinCosBeta.y;

    const double cosSigma = -(sinCosAlpha.y * sinCosDelta.y + sinCosXi.y * sinCosAlpha.x * sinCosDelta.x);
	const double cosMu = -(sinCosBeta.y * sinCosGamma.y + sinCosXi.y * sinCosBeta.x * sinCosGamma.x);
	const double cosLambda = -(sinCosAlpha.y * sinCosGamma.y - sinCosXi.y * sinCosAlpha.x * sinCosGamma.x);
    const double cosTheta = -(sinCosBeta.y * sinCosDelta.y - sinCosXi.y * sinCosBeta.x * sinCosDelta.x);

	const double q_alpha_beta = sinCosNu.x * log(tan(0.5 * alpha) * tan(0.5 * nu)) / sinCosBeta.x + sinCosNu.x * log(tan(0.5 * beta) * tan(0.5 * nu)) / sinCosAlpha.x
		                + log(tan(0.5 * alpha) * tan(0.5 * beta));

    double2 qTP_A, qTP_B;

	if ((fabs(xi) < CONSTANTS::EPS_ZERO) && (fabs(beta - gamma) < CONSTANTS::EPS_ZERO))
		qTP_A = q_thetaPsi_zero(sinCosBeta, sinCosNu, sinCosAlpha.x);
	else
		qTP_A = q_thetaPsi(sinCosAlpha, sinCosBeta, sinCosGamma, sinCosNu, sinCosXi, cosMu, cosLambda);
	
	if ((fabs(xi) < CONSTANTS::EPS_ZERO) && (fabs(alpha - delta) < CONSTANTS::EPS_ZERO))
		qTP_B = q_thetaPsi_zero(sinCosAlpha, sinCosNu, sinCosBeta.x);
	else
		qTP_B = q_thetaPsi(sinCosBeta, sinCosAlpha, sinCosDelta, sinCosNu, sinCosXi, cosSigma, cosTheta);

    res = assign_vector_part(measures[i] * (qTP_A.y * taub + qTP_B.y * taua - q_alpha_beta * tauc));
    res.w = measures[i] * (qTP_A.x + qTP_B.x);

    return res;
}

__device__ double4 integrateSingularPartSimple(int i, int j, const Point3 *vertices, const int3 *cells, const Point3 *normals, const double *measures)
{
    double4 res;

    const int3 triangleI = cells[i];
    const int3 triangleJ = cells[j];

    const int2 shifts = shiftsForSimpleNeighbors(triangleI, triangleJ);
    const int3 shiftedTriangleI = rotateLeft(triangleI, shifts.x);
    const int3 shiftedTriangleJ = rotateLeft(triangleJ, shifts.y);

    const Point3 triIA = vertices[shiftedTriangleI.x];
    const Point3 triIB = vertices[shiftedTriangleI.y];
    const Point3 triIC = vertices[shiftedTriangleI.z];
    const Point3 triJA = vertices[shiftedTriangleJ.x];
    const Point3 triJB = vertices[shiftedTriangleJ.y];
    const Point3 triJC = vertices[shiftedTriangleJ.z];

    const Point3 taua = normalize(triJA - triJC);
    const Point3 taub = normalize(triJB - triJA);

    const Point3 normalI = normals[i];
    const Point3 normalJ = normals[j];

    Point3 e = cross(normalI, normalJ);
    if(vector_length2(e) < CONSTANTS::EPS_ZERO2){
        e = taub;
        if(dot(normalI, normalJ) < 0)
            printf("Orientation is incorrect for pair (%d, %d)\n", i, j);
    } else
        e = normalize(e);

    auto getDeltas = [taua, taub, normalJ](const Point3 &e){
        double2 res;
        res.x = atan2(dot(cross(taua, e), normalJ), -dot(e, taua));
        res.y = atan2(dot(cross(e, taub), normalJ), dot(e, taub));
        return res;
    };

    double2 delta = getDeltas(e);

    if((CONSTANTS::PI - fabs(delta.x) < CONSTANTS::EPS_ZERO) || (CONSTANTS::PI - fabs(delta.y) < CONSTANTS::EPS_ZERO)){
        e *= -1;
        delta = getDeltas(e);
    }

    if((delta.x * delta.y < 0) && (fabs(delta.x - delta.y) > CONSTANTS::PI)){
        e *= -1;
        delta = getDeltas(e);
    }

    const double xi = atan2(dot(cross(normalI, normalJ), e), dot(normalI, normalJ));

    //.x is the sine value, .y is the cosine value
    double2 sinCosDeltaA, sinCosDeltaB, sinCosXi;
    sincos(delta.x, &sinCosDeltaA.x, &sinCosDeltaA.y);
    sincos(delta.y, &sinCosDeltaB.x, &sinCosDeltaB.y);
    sincos(xi, &sinCosXi.x, &sinCosXi.y);

    const Point3 s = triIC - triIB;

    const double nu = angle(triIA - triIB, triIC - triIB);
    const double mu  = angle(triIB - triIC, triIA - triIC);
    const double kappa  = angle(triIB - triIA, triIC - triIA);

    double2 sinCosMu, sinCosNu;
    sincos(mu, &sinCosMu.x, &sinCosMu.y);
    sincos(nu, &sinCosNu.x, &sinCosNu.y);

    const double logSinNu = log(sinCosNu.x);
    const double logSinMu = log(sinCosMu.x);

    const double sinKappa = sin(kappa);

    const double psi = atan2(dot(cross(e, s), normalI), dot(e, s));
    double2 sinCosPsi;
    sincos(psi, &sinCosPsi.x, &sinCosPsi.y);

    double2 sinCosNuPsi, sinCosMuPsi;
    sincos(nu + psi, &sinCosNuPsi.x, &sinCosNuPsi.y);
    sincos(mu - psi, &sinCosMuPsi.x, &sinCosMuPsi.y);

    //.x corresponds to A, .y corresponds to B
    double2 cosSigma, cosChi, cosEta, cosTheta, cosLambda;
    
    cosSigma.x = sinCosDeltaA.x * sinCosPsi.x * sinCosXi.y + sinCosDeltaA.y * sinCosPsi.y;
	cosChi.x = sinCosDeltaA.x * sinCosPsi.y * sinCosXi.y - sinCosDeltaA.y * sinCosPsi.x;
	cosEta.x = sinCosDeltaA.y * sinCosPsi.x * sinCosXi.y - sinCosDeltaA.x * sinCosPsi.y;
	cosTheta.x = sinCosDeltaA.x * sinCosNuPsi.x * sinCosXi.y + sinCosDeltaA.y * sinCosNuPsi.y;
	cosLambda.x = sinCosDeltaA.x * sinCosMuPsi.x * sinCosXi.y - sinCosDeltaA.y * sinCosMuPsi.y;

    cosSigma.y = sinCosDeltaB.x * sinCosPsi.x * sinCosXi.y + sinCosDeltaB.y * sinCosPsi.y;
	cosChi.y = sinCosDeltaB.x * sinCosPsi.y * sinCosXi.y - sinCosDeltaB.y * sinCosPsi.x;
	cosEta.y = sinCosDeltaB.y * sinCosPsi.x * sinCosXi.y - sinCosDeltaB.x * sinCosPsi.y;
	cosTheta.y = sinCosDeltaB.x * sinCosNuPsi.x * sinCosXi.y + sinCosDeltaB.y * sinCosNuPsi.y;
	cosLambda.y = sinCosDeltaB.x * sinCosMuPsi.x * sinCosXi.y - sinCosDeltaB.y * sinCosMuPsi.y;

	const double2 qTP_A = q_thetaPsi_cont(sinCosXi, mu, sinCosMu, logSinMu, logSinNu, psi, sinCosPsi, nu, sinCosNu, kappa, sinKappa, 
        sinCosNuPsi.x, sinCosMuPsi.x, delta.x, sinCosDeltaA, cosLambda.x, cosTheta.x, cosEta.x, cosSigma.x, cosChi.x);
	const double2 qTP_B = q_thetaPsi_cont(sinCosXi, mu, sinCosMu, logSinMu, logSinNu, psi, sinCosPsi, nu, sinCosNu, kappa, sinKappa, 
        sinCosNuPsi.x, sinCosMuPsi.x, delta.y, sinCosDeltaB, cosLambda.y, cosTheta.y, cosEta.y, cosSigma.y, cosChi.y);

    res = assign_vector_part(measures[i] * (qTP_A.y * taua + qTP_B.y * taub));
    res.w = (fabs(xi) < CONSTANTS::EPS_ZERO) ? 0.0 : (measures[i] * (qTP_A.x - qTP_B.x));

    return res;
}

__device__ int2 shiftsForSimpleNeighbors(const int3 &triangle1, const int3 &triangle2)
{
    int2 res;

    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            if(*(&triangle1.x + i) == *(&triangle2.x + j)){
                res.x = i;
                res.y = j;
                break;
            }

    return res;
}

__device__ int2 shiftsForAttachedNeighbors(const int3 &triangle1, const int3 &triangle2)
{
    int2 res;

    int2 adjacentVerticesIndices[2];
    int index = 0;

    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            if(*(&triangle1.x + i) == *(&triangle2.x + j)){
                adjacentVerticesIndices[index].x = i;
                adjacentVerticesIndices[index].y = j;
                ++index;
            }

    for(int i = 0; i < 3; ++i)
        if(adjacentVerticesIndices[0].x != i && adjacentVerticesIndices[1].x != i){
            res.x = i;
            break;
        }

    for(int i = 0; i < 3; ++i)
        if(adjacentVerticesIndices[0].y != i && adjacentVerticesIndices[1].y != i){
            res.y = i;
            break;
        }    

    return res;
}

__device__ int3 rotateLeft(const int3 &triangle, int shift)
{
    int3 res = { 0, 0, 0 };
    if(shift > 3)
        return res;

    if(shift == 3){
        res = triangle;
        return res;
    }

    for(int i = 0; i < 3; ++i){
        int oldIndex = i + shift;
        if(oldIndex >= 3)
            oldIndex -= 3;

        *(&res.x + i) = *(&triangle.x + oldIndex);
    }

    return res;
}

void EvaluatorJ3DK::integrateOverSimpleNeighbors()
{
    printf("\nIntegrating over simple neighbors (%d pairs)...\n", simpleNeighborsTasks.size);

    //1. Integrate the regular part numerically
    numericalIntegration(neighbour_type_enum::simple_neighbors);

    //2. Integrate the singular part analytically
    unsigned int blocks = blocksForSize(simpleNeighborsTasks.size);
    kIntegrateSingularPartSimple<<<blocks, gpuThreads>>>(simpleNeighborsTasks.size, d_simpleNeighborsIntegrals.data, simpleNeighborsTasks.data, 
                        mesh.getVertices().data, mesh.getCells().data, mesh.getCellNormals().data, mesh.getCellMeasures().data);

    //3. Convert results from the form (psi, theta) into an array of Point3 with an additional check of result
    blocks = blocksForSize(simpleNeighborsTasks.size);
    kFinalizeSimpleNeighborsResults<<<blocks, gpuThreads>>>(simpleNeighborsTasks.size, d_simpleNeighborsResults.data, d_simpleNeighborsIntegrals.data, simpleNeighborsTasks.data, mesh.getCellNormals().data, mesh.getCellMeasures().data);
}

void EvaluatorJ3DK::integrateOverAttachedNeighbors()
{
    printf("\nIntegrating over attached neighbors (%d pairs)...\n", attachedNeighborsTasks.size);

    //1. Integrate the regular part numerically
    numericalIntegration(neighbour_type_enum::attached_neighbors);

    //2. Integrate the singular part analytically
    unsigned int blocks = blocksForSize(attachedNeighborsTasks.size);
    kIntegrateSingularPartAttached<<<blocks, gpuThreads>>>(attachedNeighborsTasks.size, d_attachedNeighborsIntegrals.data, attachedNeighborsTasks.data, 
                        mesh.getVertices().data, mesh.getCells().data, mesh.getCellNormals().data, mesh.getCellMeasures().data);

    //3. Convert results from the form (psi, theta) into an array of Point3
    blocks = blocksForSize(attachedNeighborsTasks.size);
    kFinalizeNonSimpleResults<<<blocks, gpuThreads>>>(attachedNeighborsTasks.size, d_attachedNeighborsResults.data, d_attachedNeighborsIntegrals.data, attachedNeighborsTasks.data, mesh.getCellNormals().data);
}

void EvaluatorJ3DK::integrateOverNotNeighbors()
{
    printf("\nIntegrating over not neighbors (%d pairs)...\n", notNeighborsTasks.size);

    //1. Perform numerical integration
    numericalIntegration(neighbour_type_enum::not_neighbors);

    //2. Convert results from the form (psi, theta) into an array of Point3
    unsigned int blocks = blocksForSize(notNeighborsTasks.size);
    kFinalizeNonSimpleResults<<<blocks, gpuThreads>>>(notNeighborsTasks.size, d_notNeighborsResults.data, d_notNeighborsIntegrals.data, notNeighborsTasks.data, mesh.getCellNormals().data);
}

void EvaluatorJ3DK::numericalIntegration(neighbour_type_enum neighborType)
{
    const deviceVector<int3> *tasks = &getTasks(neighborType);
    const deviceVector<int3> *refinedTasks = &numIntegrator.getRefinedTasks(neighborType);

    deviceVector<int> *restTasks;
    deviceVector<double4> *integrals, *tempIntegrals;

    unsigned int blocks;

    switch (neighborType)
    {
    case neighbour_type_enum::simple_neighbors:
        restTasks = &simpleNeighborsTasksRest;
        integrals = &d_simpleNeighborsIntegrals;
        tempIntegrals = &d_tempSimpleNeighborsIntegrals;
        break;
    case neighbour_type_enum::attached_neighbors:
        restTasks = &attachedNeighborsTasksRest;
        integrals = &d_attachedNeighborsIntegrals;
        tempIntegrals = &d_tempAttachedNeighborsIntegrals;
        break;
    case neighbour_type_enum::not_neighbors:
        restTasks = &notNeighborsTasksRest;
        integrals = &d_notNeighborsIntegrals;
        tempIntegrals = &d_tempNotNeighborsIntegrals;
        break;
    default:
        return;
    }

    auto performIntegration = [neighborType, tasks](const deviceVector<int3> &refinedTasks, const Mesh3D &mesh, const NumericalIntegrator3D &numIntegrator){
        unsigned int blocks = blocksForSize(refinedTasks.size);
        switch(neighborType)
        {
        case neighbour_type_enum::simple_neighbors:
            kIntegrateRegularPartSimple<<<blocks, gpuThreads>>>(refinedTasks.size, numIntegrator.getResults(neighborType).data, refinedTasks.data, tasks->data,
                mesh.getVertices().data, mesh.getCells().data, mesh.getCellNormals().data, mesh.getCellMeasures().data,
                numIntegrator.getRefinedVertices().data, numIntegrator.getRefinedCells().data, numIntegrator.getRefinedCellMeasures().data, numIntegrator.getGaussPointsNumber());
            break;
        case neighbour_type_enum::attached_neighbors:
            kIntegrateRegularPartAttached<<<blocks, gpuThreads>>>(refinedTasks.size, numIntegrator.getResults(neighborType).data, refinedTasks.data, tasks->data,
                mesh.getVertices().data, mesh.getCells().data, mesh.getCellNormals().data, numIntegrator.getRefinedVertices().data,
                numIntegrator.getRefinedCells().data, numIntegrator.getRefinedCellMeasures().data, numIntegrator.getGaussPointsNumber());
            break;
        case neighbour_type_enum::not_neighbors:
            kIntegrateNotNeighbors<<<blocks, gpuThreads>>>(refinedTasks.size, numIntegrator.getResults(neighborType).data, refinedTasks.data, mesh.getVertices().data, mesh.getCells().data,
                numIntegrator.getRefinedVertices().data, numIntegrator.getRefinedCells().data, numIntegrator.getRefinedCellMeasures().data, numIntegrator.getGaussPointsNumber());
            break;
        default:
            break;
        }
    };

    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
        numIntegrator.resetMesh();
        unsigned int blocks = blocksForSize(refinedTasks->size);
        kFillOrdinal<<<blocks, gpuThreads>>>(refinedTasks->size, restTasks->data);
    }

    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control)
        printf("Iteration 0, integrating %d tasks\n", refinedTasks->size);
    
    //1. Perform numerical integration over pairs (i, j) of refined tasks (i - refined triangle, j - original triangle),
    //in case of automatic mesh refinement original tasks are used at this point
    performIntegration(*refinedTasks, mesh, numIntegrator);

    //2. Sum up the results of numerical integration for all triangles i
    zero_value_device(integrals->data, integrals->size);
    numIntegrator.gatherResults(*integrals, neighborType);

    if(numIntegrator.getErrorControlType() == error_control_type_enum::automatic_error_control){
        int remainingTaskCount = refinedTasks->size;
        int iteration = 0;

        numIntegrator.determineCellsToBeRefined(*restTasks, getTasks(neighborType), neighborType);

        while(remainingTaskCount && iteration < CONSTANTS::MAX_REFINE_LEVEL){
            ++iteration;

            numIntegrator.refineMesh(neighborType);
            integrals->swap(*tempIntegrals);

            refinedTasks = &numIntegrator.getRefinedTasks(neighborType);

            printf("Iteration %d, integrating %d tasks\n", iteration, refinedTasks->size);

            //Perform numerical integration over pairs of remaining refined tasks
            performIntegration(*refinedTasks, mesh, numIntegrator);
            
            //reset to zero only those integrals which have not yet converged
            blocks = blocksForSize(remainingTaskCount);
            kFillValue<double4><<<blocks, gpuThreads>>>(remainingTaskCount, integrals->data, make_double4(0, 0, 0, 0), restTasks->data);
            
            //Sum up the results of numerical integration
            numIntegrator.gatherResults(*integrals, neighborType);

            remainingTaskCount = compareIntegrationResults(neighborType, iteration == 1);

            if(remainingTaskCount){
                switch (neighborType)
                {
                case neighbour_type_enum::simple_neighbors:
                    restTasks = &simpleNeighborsTasksRest;
                    break;
                case neighbour_type_enum::attached_neighbors:
                    restTasks = &attachedNeighborsTasksRest;
                    break;
                case neighbour_type_enum::not_neighbors:
                    restTasks = &notNeighborsTasksRest;
                    break;
                }

                numIntegrator.determineCellsToBeRefined(*restTasks, getTasks(neighborType), neighborType);
            }
        }
    }
}
