#include <cstdio>
#include <cstdlib>
#include <mkl.h>

int main(int argc, char* argv[]) {

//	PRACTICA 2.1
	double v1[3] = { 2.0, 3.0, 1.0 };
	double v2[3] = { -6.0, 4.0, 0 };
	double res1;
	const MKL_INT  n1=3, incx=1, incy=1;

	res1 = cblas_ddot(n1, v1, incx, v2, incy);

	printf("			PRACTICA 2.1\n");
	printf("-----------------------------------------------------------\n");
	printf("Resultado del producto escalar: \n%1f\n\n", res1);

//	PRACTICA 2.2
	double vN[10] = {99.0,97.0,114.0,108.0,111.0,115.0,0.0,0.0,0.0,0.0};
	double vA[10] = {112.0,97.0,99.0,105.0,99.0,104.0,97.0,110.0,97.0,0.0};
	const MKL_INT n2 = 10, incn = 1, inca = 1;
	const double a = 3.0;
	
	cblas_daxpy(n2, a, vA, inca, vN, incn);
	double max = vN[cblas_idamax(n2, vN, incn)];

	printf("			PRACTICA 2.2\n");
	printf("-----------------------------------------------------------\n");
	printf("Resultado mapeado de caracteres:\n");
	for (int i = 0; i < n2; i++) {
		printf("| %c ", (int)((vN[i]/max)*25+97));
	}

//	PRACTICA 2.3
	const int n3 = 8, incf=1;
	double res3;
	double fN[] = {1.0, 8.0, 1.0, 2.0, 1.0, 9.0, 9.0, 4.0};

	res3 = fmod(cblas_dnrm2(n3, fN, incf), 11);

	printf("\n\n			PRACTICA 2.3\n");
	printf("-----------------------------------------------------------\n");
	printf("Modulo 11 de la norma 2: \n%1f", res3);

// EXTRA

	double v3[3] = { 1.0, 2.0, 3.0 };
	double v4[3] = { 3.0, 2.0, 1.0 };
	double v5[6] = { 3.0, 0, 2.0, 0, 1.0 ,0};
	double res4, res5;

	const MKL_INT  incyEx = 2;

	res4 = cblas_ddot(n1, v3, incx, v4, incy);
	res5 = cblas_ddot(n1, v3, incx, v5, incyEx);

	printf("\n\n			EXTRA\n");
	printf("-----------------------------------------------------------\n");
	printf("Producto escalar sin modificar incrementos: %1f", res4);
	printf("\nProducto escalar con incremento de y=2: %1f", res5);

	std::getchar();
	return 0;

}