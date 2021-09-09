#include <cstdio>
#include <cstdlib>
#include <mkl.h>

#include <fstream>
#include <sstream>

#include <ctime>


void load_csv(int n, int m, float* data) {
	std::ifstream file("waveform.csv");

	for (int row = 0; row < n; row++)
	{
		std::string line;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);

		for (int col = 0; col < m; col++)
		{
			std::string val;
			std::getline(iss, val, ',');
			if (!iss.good())
				break;

			std::stringstream convertor(val);
			convertor >> data[row * m + col];
		}
	}
}

int main(int argc, char* argv[]) {

	/*----- - PARTE 1------------------------------------------------------------
	Cargar los datos
	*/
	const int nRow = 5000;
	const int nColl = 10;
	float* datas = new float[nRow * nColl];
	load_csv(nRow, nColl, datas);

	/*----- - PARTE 2------------------------------------------------------------
	Centrar los datos restando la media de cada componente, generando una matriz XC
	*/
	const MKL_INT incx = 0;
	float media;
	const float a = 1;

	for (int i = 0; i < nColl; i++) {
		media = -(cblas_sdot(nRow, &datas[i], nColl, &a, incx) / (float)nRow); // Sacar media
		cblas_saxpy(nRow, a, &media, incx, &datas[i], nColl); // generando una matriz XC
	}

	/*------ - PARTE 3------------------------------------------------------------
	Calcular los autovalores y los autovectores de la matriz de covarianza -> Z = (XC’ * XC) / m
	*/

	//Matriz Covarianza
	float* datasTrans = new float[nRow * nColl];
	float* Z = new float[nColl * nColl];
	float alpha = 0.0002;

	mkl_somatcopy('R', 'T', nRow, nColl, a, datas, nColl, datasTrans, nRow);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nColl, nColl, nRow, alpha, datasTrans, nRow, datas, nColl, a, Z, nColl);

	//Mostrar Matriz Covarianza
	for (int i = 0; i < nColl * nColl; i++) {
		if (i % nColl == 0) {
			printf("\n\n | %f", Z[i]);
		}
		else {
			printf(" | %f", Z[i]);
		}
	}

	//Calcular AUTOVALORES y AUTOVECTORES

	/*const char JOBVL = 'V', JOBVR = 'V';
	const int N = nColl;
	const int LDA = N;
	float* WR = new float[N];
	float* WI = new float[N];
	float* VL = new float[N];
	const int LDVL = N;
	float* VR = new float[N];
	const int LDVR = N;
	float* WORK = new float[4 * N];
	const int LWORK = 4 * N;
	int info = 0;

	sgeev(&JOBVL, &JOBVR, &N, Z, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &info);

	printf("\n\n");
	//Mostrar Autovalores y Autovectores
	for (int i = 0; i < N * N; i++) {
		if (i % N == 0) {
			printf("\n\n    AutoValor: %f |  AutoVector: %f", WR[i / N], VL[i]);
		}
		else {
			printf(" , %f", VL[i]);
		}
	}

	/*delete[] WR;
	delete[] WI;
	delete[] VL;
	delete[] VR;
	delete[] WORK;
	delete[] Z;
	delete[] datas;
	delete[] datasTrans;*/

	std::getchar();
	return 0;
}

/*#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <cmath>
#include <ctime>

int multMatrizDouble(int lenght) {

	const int N = lenght;
	const double alphaA = 1.0;
	const double betaA = 0.0;

	const MKL_INT m = N, n = N, k = N;
	const MKL_INT lda = N, ldb = N, ldc = N;

	double* A = (double*)mkl_malloc(N * N * sizeof(double), 64);
	double* B = (double*)mkl_malloc(N * N * sizeof(double), 64);
	double* C = (double*)mkl_malloc(N * N * sizeof(double), 64);

	srand((unsigned int)time(NULL));

	for (int i = 0; i < N * N; i++) {
		A[i] = (double)rand() / (double)RAND_MAX;
		B[i] = (double)rand() / (double)RAND_MAX;
		C[i] = (double)rand() / (double)RAND_MAX;
	}

	double seconds,GFlops;
	time_t start, finish;

	time(&start);
	for (int i = 0; i < 100; i++) {
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alphaA, A, lda, B, ldb, betaA, C, ldc);
	}
	time(&finish);


	seconds = difftime(finish,start)/100;
	GFlops = (pow(N,3)*2 / seconds)*pow(10,-6);

	printf("\n\nResultados para una Matriz %i * %i Elementos",N,N);
	printf("\nTiempo Ejecucion: %1f segundos", seconds);
	printf(" - GFlops: %1f ", GFlops);

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);

	return 0;
}

int multMatrizFloat(int lenght) {

	const int N = lenght;
	const float alphaA = 1.0;
	const float betaA = 0.0;

	const MKL_INT m = N, n = N, k = N;
	const MKL_INT lda = N, ldb = N, ldc = N;

	float* A = (float*)mkl_malloc(N * N * sizeof(float), 32);
	float* B = (float*)mkl_malloc(N * N * sizeof(float), 32);
	float* C = (float*)mkl_malloc(N * N * sizeof(float), 32);

	srand((unsigned int)time(NULL));

	for (int i = 0; i < N * N; i++) {
		A[i] = (float)rand() / (float)RAND_MAX;
		B[i] = (float)rand() / (float)RAND_MAX;
		C[i] = (float)rand() / (float)RAND_MAX;
	}

	double seconds, GFlops;
	time_t start, finish;

	time(&start);
	for (int i = 0; i < 100; i++) {
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alphaA, A, lda, B, ldb, betaA, C, ldc);
	}
	time(&finish);

	seconds = difftime(finish, start) / 100;
	GFlops = (pow(N, 3) / seconds) * pow(10, -6);

	printf("\n\nResultados para una Matriz %i * %i Elementos", N, N);
	printf("\nTiempo Ejecucion: %1f segundos", seconds);
	printf(" - GFlops: %1f ", GFlops);

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);

	return 0;
}

int main(int argc, char* argv[]) {
	
	
	//Operaciones Precicion Doble
	multMatrizDouble(1500);
	multMatrizDouble(1750);
	multMatrizDouble(2000);
	multMatrizDouble(2250);
	multMatrizDouble(2500);

	//Operaciones Precicion simple
	multMatrizFloat(1500);
	multMatrizFloat(1750);
	multMatrizFloat(2000);
	multMatrizFloat(2250);
	multMatrizFloat(2500);


	std::getchar();
	return 0;
}
*/

/*printf("PRACTICA 5 \n\n");
	printf("Matriz (A): \n");
	for (int i = 0; i < N*N; i++) {
		printf("| %1f ", A[i]);
		if (fmod(i + 1, N) == 0) {
			printf("\n");
		}
	}

	printf("\nMatriz (B): \n");
	for (int i = 0; i < N*N; i++) {
		printf("| %1f ", B[i]);
		if (fmod(i + 1, N) == 0) {
			printf("\n");
		}
	}

	printf("\nMatriz Resultado:\n");
	for (int i = 0; i < N*N; i++) {
		printf("| %1f ", C[i]);
		if (fmod(i + 1, N) == 0) {
			printf("\n");
		}
	}*/