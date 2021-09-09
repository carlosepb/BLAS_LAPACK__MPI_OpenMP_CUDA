#include <cstdio>
#include <cstdlib>
#include <mkl.h>

int main(int argc, char* argv[]) {

	//	PRACTICA 3.1

	const double A[9] = { 1.0, 2.0, 3.0,
						 4.0, 5.0, 6.0,
						 7.0, 8.0, 9.0 };

	const double B[9] = { 9.0, 8.0, 7.0,
						 6.0, 5.0, 4.0,
						 3.0, 2.0, 1.0 };

	double C[9] = { 10.0, 11.0, 12.0,
						 13.0, 14.0, 15.0,
						 16.0, 17.0, 18.0 };

	const double alphaA = 1.0;
	const double betaA = 0.0;

	const MKL_INT m = 3, n = 3, k = 3;
	const MKL_INT lda = 3, ldb = 3, ldc = 3;


	//A*B

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alphaA, A, lda, B, ldb, betaA, C, ldc);

	printf("		PRACTICA 4.1 Apartado (A) \n");
	printf("-----------------------------------------------------\n");
	printf("Resultado de A*x: \n");
	for (int i = 0; i < 9; i++) {
		printf("| %.*f ", 1, C[i]);
	}

	//A* BT

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alphaA, A, lda, B, ldb, betaA, C, ldc);

	printf("\n\n		PRACTICA 4.1 Apartado (B) \n");
	printf("-----------------------------------------------------\n");
	printf("Resultado de A*x: \n");
	for (int i = 0; i < 9; i++) {
		printf("| %.*f ", 1, C[i]);
	}

	//2 * A * B + 3 * C

	double Cc[9] = { 10.0, 11.0, 12.0,
					13.0, 14.0, 15.0,
					16.0, 17.0, 18.0 };

	const double alphaC = 2.0;
	const double betaC = 3.0;
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alphaC, A, lda, B, ldb, betaC, Cc, ldc);

	printf("\n\n		PRACTICA 4.1 Apartado (C) \n");
	printf("-----------------------------------------------------\n");
	printf("Resultado de A*x: \n");
	for (int i = 0; i < 9; i++) {
		printf("| %.*f ", 1, Cc[i]);
	}

	//EXTRA
	double ME[10] = { 1.0, 2.0, 
					  1.0, 2.0,
					  1.0, 2.0, 
					  1.0, 2.0, 
					  1.0, 2.0, };


	double MR[25] = { 1.0, 1.0, 1.0, 1.0, 1.0,
					  1.0, 1.0, 1.0, 1.0, 1.0,
					  1.0, 1.0, 1.0, 1.0, 1.0, 
					  1.0, 1.0, 1.0, 1.0, 1.0,
					  1.0, 1.0, 1.0, 1.0, 1.0,};

	const MKL_INT mE = 5, nE = 5, kE = 2;
	const MKL_INT ldaE = 2, ldbE = 5, ldcE = 5;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mE, nE, kE, alphaA, ME, ldaE, ME, ldbE, betaA, MR, ldcE);

	printf("\n\n		EXTRA \n");
	printf("-----------------------------------------------------\n");
	printf("Resultado de (5x2)*(2x5): \n");
	for (int i = 0; i < 25; i++) {
		if (i % 5 != 0) {
			printf("| %.*f ", 1, MR[i]);
		}else{
			printf("\n");
			printf("| %.*f ", 1, MR[i]);		
		}
	}

	std::getchar();
	return 0;
}