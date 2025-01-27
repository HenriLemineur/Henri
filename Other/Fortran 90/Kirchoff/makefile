# Nom de l'exécutable final
EXEC = program.exe

# Dossiers
OBJ_DIR = obj

# Fichiers sources
SRC = nrtype.f90 forsythe.f90 solve.f90 decomp.f90 matrix_module.f90 main.f90
OBJ = $(SRC:.f90=.o)

# Flags de compilation
FC = gfortran
FFLAGS = -O2 -Wall

# Règle par défaut (compile tout et crée l'exécutable)
all: $(OBJ_DIR)/$(EXEC)

# Lien final pour créer l'exécutable
$(OBJ_DIR)/$(EXEC): $(addprefix $(OBJ_DIR)/, $(OBJ))
	$(FC) $(FFLAGS) -o $(OBJ_DIR)/$(EXEC) $(addprefix $(OBJ_DIR)/, $(OBJ))

# Compilation des fichiers .f90 en .o
$(OBJ_DIR)/%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Nettoyage des fichiers objets et exécutables
clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/$(EXEC) *.mod

# Règle pour exécuter le programme
run: $(OBJ_DIR)/$(EXEC)
	./$(OBJ_DIR)/$(EXEC)
