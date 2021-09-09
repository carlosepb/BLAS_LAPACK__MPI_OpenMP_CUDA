#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <cmath>
#include <random>
#include "time.h"
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
	
	int N = 4;

	double X[16] = {1, 0, 2, 0, 
					0, 1, 1, 1,
					0, 0, 4, 3,
					0, 0, 4, 3};
	
	printf("\n\n\n");
	printf("			Matrix Original");
	printf("\n-----------------------------------------------------------------");
	for (int e = 0; e < N * N; e++) {
		if (e % N == 0) {
			printf("\n | %f", X[e]);
		}
		else {
			printf(" | %f", X[e]);
		}
	}

	MKL_INT info;
	MKL_INT m = 4;
	MKL_INT n = 4;
	MKL_INT NNZ = 9;
	MKL_INT job[6] = {0, 0, 0, 2, NNZ, 1};

	double* acsr = (double*)mkl_malloc(NNZ * sizeof(double), 64);
	MKL_INT* aj = (MKL_INT*)calloc(NNZ, sizeof(MKL_INT));
	MKL_INT* ai = (MKL_INT*)calloc(m+1, sizeof(MKL_INT));

	#pragma warning(disable:4996)
	mkl_ddnscsr(job, &m, &n, X, &m, acsr, aj, ai, &info);

	printf("\n\n\n");
	printf("			Normal to CSR");
	printf("\n-----------------------------------------------------------------");

	for (int i = 0; i < NNZ; i++) {
		printf("\n | Column => %i   Value => %f", aj[i], acsr[i]);
	}

	printf("\n\n | Row offSets =>");
	for (int i = 0; i < m+1; i++) {
		printf(" %i", ai[i]);
	}

	double* acoo = (double*)mkl_malloc(NNZ * sizeof(double), 64);
	MKL_INT* ir = (MKL_INT*)calloc(NNZ, sizeof(MKL_INT));
	MKL_INT* jc = (MKL_INT*)calloc(m + 1, sizeof(MKL_INT));
	

	#pragma warning(disable:4996)
	mkl_dcsrcoo(job, &n, acsr, aj, ai, &NNZ, acoo, ir, jc, &info);

	printf("\n\n\n");
	printf("			CSR to COO");
	printf("\n-----------------------------------------------------------------");
	for (int i = 0; i < NNZ; i++) {
		printf("\n | Row => %i   Column => %i   Value => %f",ir[i], aj[i], acsr[i]);
	}

	
	//Practica 2 optativo 3

	MKL_INT mblk =  2;
	MKL_INT mBSR = 5;
	MKL_INT ldabsr = 4;

	double* absr = (double*)mkl_malloc(ldabsr * mblk * mblk *sizeof(double), 64);
	MKL_INT* jab = (MKL_INT*)calloc(ldabsr, sizeof(MKL_INT));
	MKL_INT* iab = (MKL_INT*)calloc(mBSR + 1, sizeof(MKL_INT));
	MKL_INT job1[6] = {0, 0, 0, 0, NNZ, 1};

	#pragma warning(disable:4996)
	mkl_dcsrbsr(job1, &mBSR, &mblk, &ldabsr, acsr, aj, ai, absr, jab, iab, &info);

	printf("\n\n\n");
	printf("			CSR to BSR");
	printf("\n-----------------------------------------------------------------");

	for (int i = 0; i < ldabsr * mblk * mblk; i++) {
		printf("\n | Value => %f", absr[i]);
	}

	printf("\n\n | Index of Blocks Columns =>");
	for (int i = 0; i < ldabsr; i++) {
		printf(" %i", jab[i]);
	}

	printf("\n\n | Row offSets =>");
	for (int i = 0; i < mBSR; i++) {
		printf(" %i", iab[i]);
	}
	
	
	double* acsc = (double*)mkl_malloc(NNZ * NNZ * sizeof(double), 64);
	MKL_INT* ja1 = (MKL_INT*)calloc(NNZ, sizeof(MKL_INT));
	MKL_INT* ia1 = (MKL_INT*)calloc(m + 1, sizeof(MKL_INT));

	mkl_dcsrcsc(job, &n, acsr, aj, ai, acsc, ja1, ia1, &info);

	printf("\n\n\n");
	printf("			CSR to CSC");
	printf("\n-----------------------------------------------------------------");
	for (int i = 0; i < NNZ; i++) {
		printf("\n | Row => %i   Value => %f", ja1[i], acsc[i]);
	}

	printf("\n\n | Columns offSets =>");
	for (int i = 0; i < m + 1; i++) {
		printf(" %i", ia1[i]);
	}
	printf("\n\n\n");




	//Practica 3

	/*
	std::default_random_engine generator;
	std::normal_distribution<double> rand(0.0, 1.0);
	int N = 9;

	double* A = (double*)mkl_malloc(N * N * sizeof(double), 64);
	double* B = (double*)mkl_malloc(N * N * sizeof(double), 64);
	double* C = (double*)mkl_malloc(N * N * sizeof(double), 64);

	const double alphaA = 1.0;
	const double betaA = 0.0;

	const MKL_INT m = 3, n = 3, k = 3;
	const MKL_INT lda = 3, ldb = 3, ldc = 3;


	//A*B
	for (int i = 0; i < N * N; i++) {
		A[i] = rand(generator);
		B[i] = rand(generator);
	}
	
	/*for (int e = 0; e < N * N; e++) {
			printf("\n | %f", B[e]);
	}*/
	/*
	time_t now = clock();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alphaA, A, lda, B, ldb, betaA, C, ldc);
	time_t ms = clock() - now;

	printf("Tiempo de ejecucion de la funcion Cblas_DGEMM %d ms ", ms);


	MKL_CSPBLAS_();


	printf("\n\n");
	printf("Resultado de A*x: \n");
	for (int i = 0; i < 9; i++) {
		printf("| %.*f ", 1, C[i]);
	}
	*/
}