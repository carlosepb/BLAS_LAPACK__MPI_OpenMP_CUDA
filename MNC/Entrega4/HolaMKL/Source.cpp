#include <cstdio>
#include <cstdlib>
#include <mkl.h>

int main(int argc, char* argv[]) {
	double fin, inicio, total=0.0;
	double sum1 = 300.125;
	double sum2 = 150.789;

	inicio = dsecnd();
	for (int i = 0; i < 1000000; i++) {
		sum1 * sum2;	
	}
	fin = dsecnd();
	total = fin - inicio;

	printf("Tiempo de ejecucion: %1f usec\n", (total) * 1.0e6);

	std::getchar();
	return 0;
}