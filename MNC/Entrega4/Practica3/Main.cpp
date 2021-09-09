#include <cstdio>
#include <cstdlib>
#include <mkl.h>

int main(int argc, char* argv[]) {

	//	PRACTICA 3.1

	const double x[3] = {1.0, 2.0, 3.0};
	double yA[3] = { 1.0, 2.0, 3.0};

	const double A[9] = {1.0, 2.0, 3.0,
						 4.0, 5.0, 6.0,
						 7.0, 8.0, 9.0};

	const MKL_INT m = 3, n = 3;
	const double alphaA = 1.0;
	const MKL_INT lda = 3;//preguntar
	const MKL_INT incx = 1;
	const double betaA = 0.0;
	const MKL_INT incy = 1;

	//	PRACTICA 3.1 Apartado A

	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alphaA, A
		, lda, x, incx, betaA, yA, incy);

	printf("		PRACTICA 3.1 Apartado (A) \n");
	printf("-----------------------------------------------------\n");
	printf("Resultado de A*x: \n");
	for (int i = 0; i < 3; i++) {
		printf("| %.*f ",1, yA[i]);
	}

	//	PRACTICA 3.1 Apartado B

	const double alphaB = 3.0;
	const double betaB = 4.0;

	double yB[3] = { 1.0, 2.0, 3.0 };

	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alphaB, A
		, lda, x, incx, betaB, yB, incy);

	printf("\n\n		PRACTICA 3.1 Apartado (B) \n");
	printf("-----------------------------------------------------\n");
	printf("Resultado de 3*A*x+4*y: \n");
	for (int i = 0; i < 3; i++) {
		printf("| %.*f ", 1, yB[i]);
	}

	//	PRACTICA 3.2

	double yA1[3] = { 1.0, 2.0, 3.0 };

	cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alphaA, A
		, lda, x, incx, betaA, yA1, incy);

	printf("\n\n		PRACTICA 3.2\n");
	printf("-----------------------------------------------------\n");
	printf("Resultado de A*x CblasColMajor: \n");
	for (int i = 0; i < 3; i++) {
		printf("| %.*f ", 1, yA1[i]);
	}

	double yA2[3] = { 1.0, 2.0, 3.0 };

	cblas_dgemv(CblasRowMajor, CblasTrans, m, n, alphaA, A
		, lda, x, incx, betaA, yA2, incy);

	printf("\n\nResultado de A*x CblasTrans: \n");
	for (int i = 0; i < 3; i++) {
		printf("| %.*f ", 1, yA2[i]);
	}

	//
	
	std::getchar();
	return 0;
}