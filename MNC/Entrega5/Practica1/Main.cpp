#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <cmath>
#include <ctime>

int main(int argc, char* argv[]) {

	int N = 6;

	double A[36] = { 7.8255, 2.2781, 2.8602, 2.7904, 6.8964, 0.7336,
					  2.9553, 3.2102, 6.9913, 6.7538, 1.3183, 8.2233,
					  1.5185, 8.2956, 7.9626, 9.0366, 1.2350, 7.2290,
					  8.4791, 8.2218, 4.4159, 9.0853, 1.9090, 9.2586,
					  7.8485, 5.7068, 4.4622, 7.4720, 1.4573, 4.9264,
					  2.7083, 5.7183, 4.6566, 2.6051, 5.8504, 6.5488};

	//double* A = (double*)mkl_malloc(N * N * sizeof(double), 64);
	double* A2 = (double*)mkl_malloc(N * N * sizeof(double), 64);
	lapack_int ipiv[6];
	
	srand((unsigned int)time(NULL));

	

	for (int i = 0; i < N * N; i++) {
		//A[i] = (double)rand() / (double)RAND_MAX*(double)10;
		A2[i] = A[i];
	}

	printf("\n\n\n		Matrix Aleatoria de 6x6");
	printf("\n----------------------------------------------------------------------");

	for (int e = 0; e < N*N ; e++) {
		if (e%6==0) {
			printf("\n| %f", A[e]);
		}
		else {
			printf("| %f", A[e]);
		}
	}

	printf("\n\n\n		Factorizacion LU");
	printf("\n----------------------------------------------------------------------");
	lapack_int m = 6, n = 6, lda = 6;

	LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, A, lda, ipiv);


	printf("\n");

	for (int e = 0; e < 6; e++) {
		printf("| %i", ipiv[e]);
	}

	printf("\n\n\n		Determinante");
	printf("\n----------------------------------------------------------------------");
	//Determinante (detA = detL * detU => detL = 1 y detU = {A11*A22*A33...Ann})

	double detA = 1.0;

	for (int i = 0; i < N; i+=N+1) {
		detA *= A[i];
	}
	printf("\n");
	printf(" El determinante es %f", detA);

	//Inversa AX=I
	double I[36] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
					 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
					 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
					 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
					 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

	
	lapack_int nrhs = 6, ldb = 6;

	LAPACKE_dgetrs(CblasRowMajor,'N', n, nrhs, A, lda, ipiv, I, ldb);

	printf("\n\n\n		Inversa AX=I");
	printf("\n----------------------------------------------------------------------");

	for (int e = 0; e < N * N; e++) {
		if (e % 6 == 0) {
			printf("\n| %f", I[e]);
		}
		else {
			printf("| %f", I[e]);
		}
	}

	//Inversa _dgetri()

	LAPACKE_dgetri(CblasRowMajor, n, A, lda, ipiv);

	printf("\n\n\n	Inversa con la funcion  _dgetri");
	printf("\n----------------------------------------------------------------------");

	for (int e = 0; e < N * N; e++) {
		if (e % 6 == 0) {
			printf("\n| %f", A[e]);
		}
		else {
			printf("| %f", A[e]);
		}
	}

	//PRACTICA 2

	//Inversa _dgesv()

	printf("\n\n\n		Inversa _dgesv");
	printf("\n----------------------------------------------------------------------");

	double I2[36] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
					  0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
					  0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
					  0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
					  0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
					  0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };


	LAPACKE_dgesv(CblasRowMajor, n, nrhs, A2, lda, ipiv, I2, ldb);

	printf("\n");
	for (int e = 0; e < N * N; e++) {
		if (e % 6 == 0) {
			printf("\n| %f", I2[e]);
		}
		else {
			printf("| %f", I2[e]);
		}
	}


	std::getchar();
	return 0;
}