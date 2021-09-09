clear all;
clc;

%-------PARTE 1------------------------------------------------------------
%Extraer del fichero de datos las características de tipo real. Se generará 
%una matriz X de m filas (instancias) por n columnas (dimensiones)

filename = 'waveform.csv';
M1 = readtable(filename);

table = M1{:,1:10};
[nrows,ncols] = size(table);

%-------PARTE 2------------------------------------------------------------
%Centrar los datos restando la media de cada componente, generando una matriz 
%XC
media = mean(table);
XC = zeros(nrows,ncols);
for col = 1:ncols
    XC(:,col) = table(:,col)-media(col);
end

%-------PARTE 3------------------------------------------------------------
%Calcular los autovalores y los autovectores de la matriz de covarianza 
%Z = (XC’*XC)/m
Z = (XC'*XC)./nrows;
[V,D] = eig(Z);
[d,ind] = sort(diag(D),"descend");
Ds = D(ind,ind);
Vs = V(:,ind);
%-------PARTE 4------------------------------------------------------------
%Representar los datos y los autovalores principales
dataResize = XC * Vs(:,1:2);
mediaDataResize = mean(dataResize);

figure(1)
hold on
quiver(mediaDataResize(1), mediaDataResize(2), Vs(1,1)*50, Vs(2,1)*50);
quiver(mediaDataResize(1), mediaDataResize(2), Vs(1,2)*50, Vs(2,2)*50);
quiver(mediaDataResize(1), mediaDataResize(2), Vs(1,3)*10, Vs(2,3)*10);
quiver(mediaDataResize(1), mediaDataResize(2), Vs(1,4)*5, Vs(2,4)*5);
scatter(dataResize(:,1), dataResize(:,2),".y")
title('Representacion de datos y autovectores')
%-------PARTE 5------------------------------------------------------------
%¿Qué ocurre al multiplicar los datos por la matriz de autovectores?
figure(2)

hold on;
final = table*Vs;
final(1:5, :)
final(2501:2505, :)
final(4995:5000, :)

scatter(final(1:2000,1), table(1:2000,1),".b")
scatter(final(1:2000,2), table(1:2000,2),".r")
scatter(final(1:2000,3), table(1:2000,3),".m")
scatter(final(1:2000,4), table(1:2000,4),".g")
scatter(final(1:2000,5), table(1:2000,5),".y")
title('Datos proyectados segun la matriz de autovectores')











