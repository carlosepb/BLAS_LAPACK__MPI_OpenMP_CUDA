#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <cmath>
#include <random>
#include "time.h"

int copyMatrix(double* A, double* B, int length) {
	for (int i = 0; i < length; i++) {
		A[i] = B[i];
	}
	return 0;
}

int identityMatrix(double* A, int rows, int cols) {
	for (int i = 0; i < rows*cols; i++) {
		if (i % (rows+1) == 0) {
			A[i] = 1;
		}
		else {
			A[i] = 0;
		}
	}
	return 0;
}

int ceroMatrix(double* matrix, int length) {
	for (int i = 0; i < length ; i++) {
		matrix[i] = 0;
	}
	return 0;
}

int addBand(double* matrix, int bandPosition, double* matrixResult, int matrixResultY, int matrixResultRow) {

	if(bandPosition>0){
		for (int i = 0; i < matrixResultY; i++) {
			matrixResult[matrixResultRow* matrixResultY+ i + bandPosition] = matrix[i* (matrixResultY+1)+1];
		}
	}
	else if (bandPosition < 0) {
		bandPosition *= -1;
		for (int i = 0; i < matrixResultY; i++) {
			matrixResult[matrixResultRow * matrixResultY + i] = matrix[bandPosition*matrixResultY+i*(matrixResultY+1)];
		}
	}
	else {
		for (int i = 0; i < matrixResultY; i++) {
			matrixResult[matrixResultRow * matrixResultY + i] = matrix[i * (matrixResultY + 1)];
		}
	}
	return 0;
}

int main(int argc, char* argv[]) {

	std::default_random_engine generator;
	std::normal_distribution<double> rand(0.0,1.0);

	int N = 6;

	double* A = (double*)mkl_malloc(N * N * sizeof(double), 64);
	double* B = (double*)mkl_malloc(N * N * sizeof(double), 64);
	double* B_copy = (double*)mkl_malloc(N * N * sizeof(double), 64);
	
	//Generar matriz de banda (A) y matriz normal (B) 

	for (int i = 0; i < N * N; i++) {
		if ((i-1)%(N+1)==0 || i%(N+1)==0 || (i+1)%(N+1)==0 ) {
			A[i] = rand(generator);
		}
		else {
			A[i] = 0.0;
		}
			B[i] = rand(generator);
			B_copy[i] = B[i];
	}

	printf("\n\n");
	for (int e = 0; e < N * N; e++) {
		if (e % 6 == 0) {
			printf("\n | %f", A[e]);
		}
		else {
			printf(" | %f", A[e]);
		}
	}

	printf("\n\n");
	for (int e = 0; e < N * N; e++) {
		if (e % 6 == 0) {
			printf("\n| %f", B[e]);
		}
		else {
			printf("| %f", B[e]);
		}
	}

	//Codificar Martriz A (A_banda)
	int N_banda_x = 6;
	int N_banda_y = 4;

	double* A_banda = (double*)mkl_malloc(N_banda_x * N_banda_y * sizeof(double), 64);
	double* A_banda_copy = (double*)mkl_malloc(N_banda_x * N_banda_y * sizeof(double), 64);


	for (int i = 0; i < 24;i++) {
		A_banda[i] = 0;
	}

	ceroMatrix(A_banda, N_banda_x * N_banda_y);

	addBand(A, 1, A_banda, 6 , 1);
	addBand(A, 0, A_banda, 6, 2);
	addBand(A, -1, A_banda, 6, 3);

	copyMatrix(A_banda_copy, A_banda, 24);

	printf("\n\n");
	for (int e = 0; e < N_banda_x * N_banda_y; e++) {
		if (e % 6 == 0) {
			printf("\n| %f", A_banda[e]);
		}
		else {
			printf("| %f", A_banda[e]);
		}
	}
	
	printf("\n\n");
	int ipiv[6];
	double I[36] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
					 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
					 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
					 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };


	time_t now = clock();
	int time_DGESV = 0;
	for (int i = 0; i < 1000000; i++) {
		copyMatrix(B, B_copy, N * N);
		identityMatrix(I, 6, 6);
		time_DGESV = LAPACKE_dgesv(CblasRowMajor, N, 6, A, N, ipiv, I, 6);
	}
	time_t ms = clock() - now;
	printf("Tiempo de ejecucion de la funcion DGESV %d ms valor de retorno de la funcion %i", ms, time_DGESV);


	printf("\n\n");
	int ipiv2[6];
	double I2[36] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
					 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
					 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
					 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

	time_t now1 = clock();
	int time_DGBSV = 0;
	for (int i = 0; i < 1000000; i++) {
		copyMatrix(A_banda, A_banda_copy, 24);
		identityMatrix(I2, 6, 6);
		time_DGBSV = LAPACKE_dgbsv(CblasRowMajor, 6, 1, 1, 6, A_banda, 6, ipiv2, I2, 6);
	}
	time_t ms1 = clock() - now1;
	printf("Tiempo de ejecucion de la funcion DGBSV %d ms valor de retorno de la funcion %i", ms1, time_DGBSV);

	std::getchar();
	return 0;
}