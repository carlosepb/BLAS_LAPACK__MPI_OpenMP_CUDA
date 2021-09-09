#include <omp.h>
#include <cstdio>

//-----------PRACTICA-6---------------------------------------------------

/*
void main()
{
	printf("[COMIENZO]\n");

	const int length = 30000;

	double a[length], b[length], c[length], d[length];

	for (int i = 0; i < length; i++) {
		a[i] = i;
		b[i] = i * (double)2;
		c[i] = a[i] + b[i];
		d[i] = a[i] * b[i];
	}

	#pragma omp parallel num_threads(4)
	{
		int id = omp_get_thread_num();
		#pragma omp sections
		{
			#pragma omp section
			{
				printf("Hilo %d Realizando Suma Primera Mitad\n", id);
				for (int i = 0; i < length / 2; i++) {
					c[i] = a[i] + b[i];
				}
			}

			#pragma omp section
			{
				printf("Hilo %d Realizando Suma Segunda Mitad\n", id);
				for (int i = length / 2; i < length; i++) {
					c[i] = a[i] + b[i];
				}
			}

			#pragma omp section 
			{
				printf("Hilo %d Realizando Multiplicacion Primera Mitad\n", id);
				for (int i = 0; i < length / 2; i++) {
					d[i] = a[i] * b[i];
				}
			}

			#pragma omp section 
			{
				printf("Hilo %d Realizando Multiplicacion Segunda Mitad\n", id);
				for (int i = length / 2; i < length; i++) {
					d[i] = a[i] * b[i];
				}
			}
		}
	}

	printf("\n[FINAL]\n");

	std::getchar();
	return;
}
*/

//-----------PRACTICA-1---------------------------------------------------
/*
void main()
{
	printf("[COMIENZO]\n");

	#pragma omp parallel num_threads(4)
	{
		int id = omp_get_thread_num();
		int num = omp_get_num_threads();

		if(id==0)
		{
			printf("Hola desde el hilo %d somos %d\n", id, num);
		}else	
		printf("Hola desde el hilo %d \n", id);
	}

	printf("[FINAL]\n");

	std::getchar();
	return;
}*/


//-----------PRACTICA-2---------------------------------------------------
/*
void main()
{
	printf("[COMIENZO]\n");

	#pragma omp parallel num_threads(9)
	{
		double a = 1.5, b = 2, c = 0;

		int id = omp_get_thread_num();
		int num = omp_get_num_threads();

		double tinicial = omp_get_wtime();
		for (int i = 0; i < 3000000; i++) {
			c += a * b-(b-a)/b;
		}
		double tfinal = omp_get_wtime();
		double tiempo = tfinal - tinicial;

		if(id==0)
		{
			printf("Hola desde el hilo %d - tiempo %1f - calculo %1f, somos %d\n", id, tiempo, c, num);
		}else
		printf("Hola desde el hilo %d - tiempo %1f - calculo %1f\n", id, tiempo, c);
	}

	printf("[FINAL]\n");

	std::getchar();
	return;
}*/


//-----------PRACTICA-3---------------------------------------------------
/*
void main()
{
	printf("[COMIENZO]\n");

	double a[100], b[100], c[100], d[100];

	for (int i = 0; i < 100; i++) {
		a[i] = i;
		b[i] = i * (double)2;
		c[i] = a[i] + b[i];
	}

	#pragma omp parallel num_threads(4)
	{
		int id = omp_get_thread_num();

		#define CHUNK 10
		#pragma omp for schedule(dynamic, CHUNK)
		for (int i = 0; i < 100; i++) {
			d[i] = a[i] + b[i];
			if (i<10) {
				printf("\nHilo: %d - Posicion: %d  - Resultado %1f", id, i, d[i]);
			}
			else {
				printf("\nHilo: %d - Posicion: %d - Resultado %1f", id, i, d[i]);
			}

		}
	}


	printf("\n\n");

	for (int i = 0; i < 100; i++) {
		if (i % 10 == 0) {
			printf("\n | %1f ", c[i]);
		}
		else {
			printf(" | %1f ", c[i]);
		}
	}

	printf("\n\n[FINAL]\n");

	std::getchar();
	return;
}
*/
//-----------PRACTICA-4---------------------------------------------------

void main()
{
	printf("[COMIENZO]\n");

	const int length = 20000;
	double tinicial, tfinal, tiempoPar;

	double a[length], b[length], c[length], d[length];

	for (int i = 0; i < length; i++) {
		a[i] = i;
		b[i] = i * (double)2;
		c[i] = a[i] + b[i];
	}

	tinicial = omp_get_wtime();
	#pragma omp parallel num_threads(4)
	{
		//int id = omp_get_thread_num();

		#define CHUNK 1000
		#pragma omp for schedule(static, CHUNK)
		for (int i = 0; i < length; i++) {
			d[i] = a[i] + b[i];
			/*if (i<10) {
				printf("\nH->%d  P->%d   R->%1f", id, i, d[i]);
			}
			else {
				printf("\nH->%d  P->%d  R->%1f", id, i, d[i]);
			}*/
		}
	}
	tfinal = omp_get_wtime();
	tiempoPar = tfinal - tinicial;

	printf("\nTiempo Paralelo: %1f", tiempoPar);

	printf("\n\n[FINAL]\n");

	std::getchar();
	return;
}



//-----------PRACTICA-5---------------------------------------------------
/*
void main()
{
	printf("[COMIENZO]\n");

	const int length = 100;

	double a[length], b[length], c[length], d[length];

	for (int i = 0; i < length; i++) {
		a[i] = i;
		b[i] = i * (double)2;
		c[i] = a[i] + b[i];
		d[i] = a[i] * b[i];
	}

	#pragma omp parallel num_threads(4)
	{
		int id = omp_get_thread_num();
		#pragma omp sections
		{
			#pragma omp section
			{
				//printf("Hilo %d Realizando Suma vectores\n", id);
				for (int i = 0; i < length; i++) {
					c[i] = a[i] + b[i];
					printf("\nSUM -> H->%d  P->%d   R->%1f", id, i, c[i]);
				}
			}
			#pragma omp section 
			{
				//printf("\nHilo %d Realizando Multiplicacion vectores\n", id);
				for (int i = 0; i < length; i++) {
					d[i] = a[i] * b[i];
					printf("\nMUL -> H->%d  P->%d   R->%1f", id, i, d[i]);
				}
			}
		}
	}

	printf("\n\n[FINAL]\n");

	std::getchar();
	return;
}
*/