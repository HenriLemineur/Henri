EXECUTABLE=main.exe
SRC=nrtype.f90 forsythe.f90 quanc8.f90 modules.f90 main.f90

OBJ = $(SRC:.f90=.o)

$(EXECUTABLE): $(OBJ)
	gfortran $^ -o $@

%.o: %.f90
	gfortran -c $< -o $@

clean:
	rm -f *.o *.mod *.dat $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE)
	python plot.py