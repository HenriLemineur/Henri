#PROJECTS_ROOT=C:\Users\hlemineu\OneDrive - Université de Namur\Documents\Master\Master 2\Methode num\TD\TD-4_decompSCV
OPENBLAS_LOCATION=openblas\lib\libopenblas.a
OPENBLAS_NUM_THREADS=1
FC_COMPILE_OPTIONS=-m32 -O3 -fcheck=all
FC_LINK_OPTIONS=${OPENBLAS_LOCATION} -m32 -flto

main.exe: main.o 
	gfortran main.o -o main.exe ${FC_LINK_OPTIONS}

main.o: main.f90
	gfortran -c main.f90 -o main.o ${FC_COMPILE_OPTIONS}

plot:
	python -m Plot


clean:
	rm *.o
	rm main.exe
	rm *.jpg
	rm Decompdat.dat
	rm Initdat.dat
