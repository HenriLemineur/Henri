EXECUTABLE=main.exe
SRC=nrtype.f90 nr.f90 donnees_communes.f90 outils.f90 nrutil.f90 locate.f90 tridag.f90 splint.f90 spline.f90 golden.f90 main.f90

OBJ = $(SRC:.f90=.o)

$(EXECUTABLE): $(OBJ)
	gfortran $^ -o $@

%.o: %.f90
	gfortran -c $< -o $@

clean:
	rm -f *.o *.mod $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE)