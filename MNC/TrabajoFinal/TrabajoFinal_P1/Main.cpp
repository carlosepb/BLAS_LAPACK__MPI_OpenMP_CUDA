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
	float* datas = new float[nRow*nColl];
	load_csv(nRow, nColl, datas);

	/*----- - PARTE 2------------------------------------------------------------
	Centrar los datos restando la media de cada componente, generando una matriz XC
	*/
	const MKL_INT incx = 0;
	float media;
	const float a = 1;

	for (int i = 0; i < nColl; i++) {
		media = -(cblas_sdot(nRow, &datas[i], nColl, &a, incx)/(float)nRow); // Sacar media
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
	printf("\n\n   Matriz de Covarianza\n");
	for (int i = 0; i < nColl * nColl; i++) {
		if (i % nColl == 0) {
			printf("\n\n | %f", Z[i]);
		}
		else {
			printf(" | %f", Z[i]);
		}
	}

	//Calcular AUTOVALORES y AUTOVECTORES
	
	const char JOBVL = 'V', JOBVR = 'N';
	const int N = nColl;
	const int LDA = N;
	float* WR = new float[N];
	float* WI = new float[N];
	float* VL = new float[N];
	const int LDVL = N;
	float* VR = new float[N];
	const int LDVR = N;
	float* WORK = new float[4*N];
	const int LWORK = 4*N;
	int info = 0;

	sgeev(&JOBVL, &JOBVR, &N, Z, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &info);

	printf("\n\n   Matriz De Autovalores\n");
	  
	for (int i = 0; i < nColl * nColl; i++) {
		if (i % nColl == 0) {
			printf("\n\n | %f", Z[i]);
		}
		else {
			printf(" | %f", Z[i]);
		}
	}

	printf("\n\n   Auto Vectores\n");

	for (int i = 0; i < N * N; i++) {
		if (i % N == 0) {
			printf("\n\n |  AutoVector: %f", VL[i]);
		}
		else {
			printf(" , %f", VL[i]);
		}
	}
	//Mostrar Autovalores y Autovectores
	/*for (int i = 0; i < N*N; i++) {
		if (i%N==0) {
			printf("\n\n    AutoValor: %f |  AutoVector: %f", WR[i/N], VL[i]);
		}
		else {
			printf(" , %f", VL[i]);
		}
	}*/

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