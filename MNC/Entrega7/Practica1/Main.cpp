#include "mpi.h"
#include <stdio.h>

/*
Para ejecutar:
	-abrir CMD
	-utizar comando: cd source\repos\Entrega7\Practica1\x64\Release
	-ejecutar practica: mpiexec -n "numeroProcesos" practica1.exe
*/


//----------------PRACTICA 1-------------------------------------------------------------------------

/*
Ejecutar:
	- mpiexec -n 4 practica1.exe
	- mpiexec -n 4 practica1.exe
	- mpiexec -n 4 practica1.exe
Explicar que como vemos los mensajes no aparecen en el mismo orden en las
diferentes ejecuciones debido a que no es determinista y los mensajes se 
generan en procesos diferentes.
*/

/*
void main(int argc, char* argv[])
{
	int rank, size, length;
	char name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Get_processor_name(name, &length);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("[Proceso %d] Hola desde %s!\n", rank, name);
	if (rank == 0) printf("El numero total de procesos MPI es % d.\n", size);
	MPI_Finalize();
	return;
}*///----------------PRACTICA 2------------------------------------------------------------------------

/*

Ejecutar:
	- mpiexec -n 7 practica1.exe
	- mpiexec -n 8 practica1.exe
	- mpiexec -n 9 practica1.exe
Explicar como se produce una varianza clara en los tiempos entre la ejecucion de -n 8 y -n 9
mientras que entre -n 7 y -n 8 no hay ningun cambio por lo que podemos deducir que el numero
maximo de procesos simultaneos es de 8. Tambien decir que imprimimos el resultado debido a 
que si no realizamos ninguna accion con el valor que se genera de las operacion el compilador
para optimizar la ejecucion omite la parte del codigo donde se trabaja con esto ya que al
ver que no tiene uso alguno lo considera innecesario.
Si los tiempos no salen como deberian repetir la ejecucion hasta que salga como se debe
ya que puede haber ruido de otros porcesos.
*/

/*
void main(int argc, char* argv[])
{
	int rank, size, length;
	char name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Get_processor_name(name, &length);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double a = 1.456, b = 1.321, c=0;

	double tinicial = MPI_Wtime();
	for (int i = 0; i < 10000000; i++) {
		c += a * b *a *b *a *b;
	}
	double tfinal = MPI_Wtime();
	double tiempo = tfinal - tinicial;

	printf("[Proceso %d] tiempo => %f	resultado => %f\n", rank, tiempo, c);
	//printf("[Proceso %d] Hola desde %s!\n", rank, name);
	//if (rank == 0) printf("El numero total de procesos MPI es % d.\n", size);
	MPI_Finalize();
	return;
}*///----------------PRACTICA 3------------------------------------------------------------------------
/*
Ejecutar:
	- mpiexec -n 4 practica1.exe
	- mpiexec -n 5 practica1.exe
Explicar que como podemos ver por el resultado cada proceso envia su identificador a otro
y recibe de este el suyo, ademas ya que los procesos se organizaban por parejas adaptamos
el codigo para que en el caso de que reciba un numero de procesos impar sea el proceso 0 
el que se ocupe del sobrante.
*/

/*
void main(int argc, char* argv[])
{
	MPI_Status status;
	int partner, mensaje;

	int rank, size, length;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	if ((size % 2) == 1 && rank == (size-1)) {
		partner = 0;
	}
	else if (rank < (size/2)) {
		partner = rank + size / 2;
	}
	else {
		partner = rank - size / 2;
	}

	if ((size%2) == 1 && rank == 0) {
		MPI_Send(&rank, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
		MPI_Recv(&mensaje, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
		printf("[Proceso %d] | mensaje recibido: %d | procedente de: [Proceso %d]\n", rank, mensaje, partner);

		partner = size - 1;

		MPI_Send(&rank, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
		MPI_Recv(&mensaje, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
		printf("[Proceso %d] | mensaje recibido: %d | procedente de: [Proceso %d]\n", rank, mensaje, partner);
	}
	else {
		MPI_Send(&rank, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
		MPI_Recv(&mensaje, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
		printf("[Proceso %d] | mensaje recibido: %d | procedente de: [Proceso %d]\n", rank, mensaje, partner);
	}

	MPI_Finalize();
	return;
}
*/

//----------------PRACTICA 4------------------------------------------------------------------------

/*
Ejecutar:
	- mpiexec -n 6 practica1.exe
Explicar que modificamos los metodos de comunicacion y añadimos la funcion wait y que como
vemos por la ejecucion tenemos el mismo resultado.
*/


/*
void main(int argc, char* argv[])
{
	MPI_Status status[2];
	MPI_Request request[2];
	int partner, mensaje;

	int rank, size, length;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	if (rank < (size/2)) {
		partner = rank + size / 2;
	}
	else {
		partner = rank - size / 2;
	}

	MPI_Isend(&rank, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &request[1]);
	MPI_Irecv(&mensaje, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &request[0]);
	MPI_Waitall(2, request, status);

	printf("[Proceso %d] | mensaje recibido: %d | procedente de: [Proceso %d]\n", rank, mensaje, partner);

	MPI_Finalize();
	return;
}

*/

//----------------PRACTICA 5------------------------------------------------------------------------

/*
Ejecutar:
	- mpiexec -n 6 practica1.exe
Explicar que cambiamos el tipo de dato que se va a enviar que era un entero por un
array de caracteres, modificando la variable, el tipo de datos y el tamaño en las 
funciones de enviar y recibir.
Para crear la cadena que se va a enviar usamos la funcion sprint_s para concatenar
los caracteres de esta con el identificador del proceso.
*/


/*
void main(int argc, char* argv[])
{
	MPI_Status status[2];
	MPI_Request request[2];

	int partner;
	char envio[15], mensaje[15];

	int rank, size, length;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank < (size/2)) {
		partner = rank + size / 2;
	}
	else {
		partner = rank - size / 2;
	}

	sprintf_s(envio, 15, "Emisor %d", rank);

	MPI_Isend(&envio, 15, MPI_CHAR, partner, 1, MPI_COMM_WORLD, &request[1]);
	MPI_Irecv(&mensaje, 15, MPI_CHAR, partner, 1, MPI_COMM_WORLD, &request[0]);
	MPI_Waitall(2, request, status);

	printf("[Proceso %d] | mensaje recibido: %s | procedente de: [Proceso %d]\n", rank, mensaje, partner);

	MPI_Finalize();
	return;
}
*/

//----------------PRACTICA 6------------------------------------------------------------------------

/*
Ejecutar:
	- mpiexec -n 8 practica1.exe
Explicar que podemos ver los mensajes recibidos por los distintos procesos para las diferentes
formas de enviarlos (punto a punto o broadcast) y el tiempo total que tardaron estos en recibirlos
y que como analizamos de los datos de los tiempos la funcion broadcast comparada con funciones 
punto a punto bloqueantes es mas rapida.
*/

/*
void main(int argc, char* argv[])
{
	MPI_Status status;
	MPI_Request request[2];

	int rank, size, length;
	double tinicial, tfinal, tiempo;
	char buffer1[102400];
	char buffer2[102400];
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	//-------------------PUNTO A PUNTO---------------------------------
	if (rank == 0) {
		sprintf_s(buffer1, 102400, "Texto largo para prueba... Punto a punto!");
		tinicial = MPI_Wtime();
		for (int i = 1; i < size; i++) {
			MPI_Send(&buffer1, 102400, MPI_CHAR, i, 1, MPI_COMM_WORLD);
		}
		tfinal = MPI_Wtime();
		tiempo = tfinal - tinicial;
		printf("[Proceso %d] | Tiempo punto a punto: %f \n", rank, tiempo);
	}
	else {
		MPI_Recv(&buffer1, 102400, MPI_CHAR, 0, 1, MPI_COMM_WORLD,&status);
		printf("[Proceso %d] | mensaje recibido: %s \n", rank, buffer1);
	}

	//-------------------BROADCAST---------------------------------
	if (rank == 0) {
		sprintf_s(buffer2, 102400, "Texto largo para prueba... Broadcast!");
		tinicial = MPI_Wtime();
	}

	MPI_Bcast(&buffer2, 102400, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		tfinal = MPI_Wtime();
		tiempo = tfinal - tinicial;
		printf("[Proceso %d] | Tiempo broadcast: %f \n", rank, tiempo);
	}
	else {
		printf("[Proceso %d] | mensaje recibido: %s \n", rank, buffer2);
	}

	MPI_Finalize();
	return;
}
*/

//----------------PRACTICA 7------------------------------------------------------------------------

/*
Ejecutar:
	- mpiexec -n 2 practica1.exe
	- mpiexec -n 4 practica1.exe
Explicar que podemos ver como el calculo total en serie y en paralelo dan el mismo resultado
ademas de los calculos de los que se encarga cada hilo por separa antes de devolverlos al 
proceso principal para el resultado en paralelo.
*/

/*
void main(int argc, char* argv[])
{
	MPI_Status status[2];
	MPI_Request request[2];

	int rank, size, length;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int vectorLength = 10000;
	const int vectorCutLength = vectorLength / size;

	double vector[vectorLength];
	double vectorCut[vectorLength];
	double vectorAcumulados[100];

	double total = 0.0;

	if (rank == 0) {
		double valSerie = 0.0;
		for (int i = 0; i < vectorLength; i++) {
			vector[i] = (double)1.0;
		}

		for (int i = 0; i < vectorLength; i++) {
			for (int j = 0; j < vectorLength; j++) {
				valSerie += vector[i];
			}
		}
		printf("[Proceso %d] | Total Serie: %f\n", rank, valSerie);
	}

	MPI_Scatter(&vector, vectorCutLength, MPI_DOUBLE, &vectorCut, vectorCutLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	double val = 0.0;
	for (int i = 0; i < vectorCutLength; i++) {
		for (int j = 0; j < vectorLength; j++) {
			val += vectorCut[i];
		}
	}
	printf("[Proceso %d] | Suma Hilo: %f\n", rank, val);

	MPI_Gather(&val, 1, MPI_DOUBLE, &vectorAcumulados, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		for (int i = 0; i < size; i++) {
			total += vectorAcumulados[i];
		}
		printf("[Proceso %d] | Total Paralelo: %f\n", rank, total);
	}
	MPI_Finalize();
	return;
}
*/

//----------------PRACTICA 8------------------------------------------------------------------------

/*
Ejecutar:
	- mpiexec -n 2 practica1.exe
	- mpiexec -n 4 practica1.exe
	- mpiexec -n 8 practica1.exe
	- mpiexec -n 16 practica1.exe

Comentar que esta vez solo imprimimos el resultado total para cada una de las maneras de 
calcular(Paralela y serie) debido a que utilizaremos multiples hilos y no queremos saturar
la salida de resultados para las multiples pruebas que haremos.
De los resultados podemos sacar en claro que como vemos al doblar el numero de procesos
el tiempo de ejecucion en paralelo se va reduciendo en aproximadamente la mitad hasta 
llegar al numero maximo de procesos simultaneos que soporta fisicamente el equipo, en
este caso 8.

cambiar vectorLength=80 (linea 407)

Ejecutar:
	- mpiexec -n 2 practica1.exe
	- mpiexec -n 4 practica1.exe
	- mpiexec -n 8 practica1.exe

Como vemos en los resultados el tiempo de ejecucion de paralelo es mayor que en serie
consiguiendo cada vez peores resultados segun aumentamos los hilos esto es debido a
que para que sea optimo utilizar multiples procesos el volumen de datos tienen que
ser mucho mayor y al disminuir el numero de elementos del vector no podemos aprovechar
la ejecucion concurrente.

*/

/*
void main(int argc, char* argv[])
{
	MPI_Status status[2];
	MPI_Request request[2];

	int rank, size, length;
	double tinicial, tfinal, tiempo;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int vectorLength = 10000;
	const int vectorCutLength = vectorLength / size;

	double vector[vectorLength];
	double vectorCut[vectorLength];
	double vectorAcumulados[100];
	
	double total = 0.0;

	if (rank == 0 ) {
		double valSerie = 0.0;
		for (int i = 0; i < vectorLength; i++) {
			vector[i] = (double)1.0;
		}
		tinicial = MPI_Wtime();
		for (int i = 0; i < vectorLength; i++) {
			for (int j = 0; j < 10000; j++) {
				valSerie += vector[i];
			}
		}
		tfinal = MPI_Wtime();
		tiempo = tfinal - tinicial;

		printf("---------------------------------------\n");
		printf("Numero de elementos del Vector: %i\n", vectorLength);
		printf("---------------------------------------\n");
		printf("[Proceso %d] | Total Serie:    %f | Tiempo Serie:    %f\n", rank, valSerie, tiempo);

		tinicial = MPI_Wtime();
	}

	MPI_Scatter(&vector, vectorCutLength, MPI_DOUBLE, &vectorCut, vectorCutLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	double val = 0.0;
	for (int i = 0; i < vectorCutLength; i++) {
		for (int j = 0; j < 10000; j++) {
			val += vectorCut[i];
		}
	}
	//printf("[Proceso %d] | Suma Hilo: %f\n", rank, val);

	MPI_Gather(&val, 1, MPI_DOUBLE, &vectorAcumulados, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		for (int i = 0; i < size; i++) {
			total += vectorAcumulados[i];
		}
		tfinal = MPI_Wtime();
		tiempo = tfinal - tinicial;

		printf("[Proceso %d] | Total Paralelo: %f | Tiempo Paralelo: %f\n", rank, total, tiempo);
	}
	MPI_Finalize();
	return;
}*/


//----------------PRACTICA 9------------------------------------------------------------------------

/*
Ejecutar:
	- mpiexec -n 2 practica1.exe
	- mpiexec -n 4 practica1.exe
Explicar que podemos ver como el calculo total en serie y en paralelo dan el mismo resultado
ademas de los calculos de los que se encarga cada hilo por separa antes de devolverlos al
proceso principal para el resultado en paralelo.
*/

/*
void main(int argc, char* argv[])
{
	int rank, size, length;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int vectorLength = 10000;
	const int vectorCutLength = vectorLength / size;

	double vector[vectorLength];
	double vectorCut[vectorLength];
	double vectorAcumulados[100];
	double val = 0.0;
	double total = 0.0;

	if (rank == 0) {
		double valSerie = 0.0;
		for (int i = 0; i < vectorLength; i++) {
			vector[i] = (double)1.0;
		}

		for (int i = 0; i < vectorLength; i++) {
			for (int j = 0; j < vectorLength; j++) {
				valSerie += vector[i];
			}
		}
		printf("[Proceso %d] | Total Serie:    %f\n", rank, valSerie);
	}

	MPI_Scatter(vector, vectorCutLength, MPI_DOUBLE, vectorCut, vectorCutLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i < vectorCutLength; i++) {
		for (int j = 0; j < vectorLength; j++) {
			val += vectorCut[i];
		}
	}
	printf("[Proceso %d] | Suma Hilo: %f\n", rank, val);

	MPI_Reduce(&val, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("[Proceso %d] | Total Paralelo: %f\n", rank, total);
	}
	MPI_Finalize();
	return;
}
*/

//----------------PRACTICA 10-----------------------------------------------------------------------

/*
Ejecutar:
- mpiexec - n 2 practica1.exe
- mpiexec - n 4 practica1.exe
- mpiexec - n 8 practica1.exe
- mpiexec - n 16 practica1.exe

Comentar que esta vez solo imprimimos el resultado total para cada una de las maneras de
calcular(Paralela y serie) debido a que utilizaremos multiples hilos y no queremos saturar
la salida de resultados para las multiples pruebas que haremos.
De los resultados podemos sacar en claro que como vemos al doblar el numero de procesos
el tiempo de ejecucion en paralelo se va reduciendo en aproximadamente la mitad hasta
llegar al numero maximo de procesos simultaneos que soporta fisicamente el equipo, en
este caso 8.

cambiar vectorLength = 80 (linea 407)

Ejecutar:
- mpiexec - n 2 practica1.exe
- mpiexec - n 4 practica1.exe
- mpiexec - n 8 practica1.exe

Como vemos en los resultados el tiempo de ejecucion de paralelo es mayor que en serie
consiguiendo cada vez peores resultados segun aumentamos los hilos esto es debido a
que para que sea optimo utilizar multiples procesos el volumen de datos tienen que
ser mucho mayor y al disminuir el numero de elementos del vector no podemos aprovechar
la ejecucion concurrente.
*/


void main(int argc, char* argv[])
{
	int rank, size, length;
	double tinicial, tfinal, tiempo;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const int vectorLength = 10000;
	const int vectorCutLength = vectorLength / size;

	double vector[vectorLength];
	double vectorCut[vectorLength];
	double vectorAcumulados[100];
	double val = 0.0;
	double total = 0.0;

	if (rank == 0 ) {
		double valSerie = 0.0;
		for (int i = 0; i < vectorLength; i++) {
			vector[i] = (double)1.0;
		}
		tinicial = MPI_Wtime();
		for (int i = 0; i < vectorLength; i++) {
			for (int j = 0; j < 10000; j++) {
				valSerie += vector[i];
			}
		}
		tfinal = MPI_Wtime();
		tiempo = tfinal - tinicial;

		printf("---------------------------------------\n");
		printf("Numero de elementos del Vector: %i\n", vectorLength);
		printf("---------------------------------------\n");
		printf("[Proceso %d] | Total Serie:    %f | Tiempo Serie:    %f\n", rank, valSerie, tiempo);

		tinicial = MPI_Wtime();
	}

	MPI_Scatter(vector, vectorCutLength, MPI_DOUBLE, vectorCut, vectorCutLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i < vectorCutLength; i++) {
		for (int j = 0; j < 10000; j++) {
			val += vectorCut[i];
		}
	}
	//printf("[Proceso %d] | Suma Hilo: %f\n", rank, val);

	MPI_Reduce(&val, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		tfinal = MPI_Wtime();
		tiempo = tfinal - tinicial;
		printf("[Proceso %d] | Total Paralelo: %f | Tiempo Paralelo: %f\n", rank, total, tiempo);
	}

	MPI_Finalize();
	return;
}

