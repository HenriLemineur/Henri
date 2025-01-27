# Nom du fichier principal
MAIN = main.tex

# Fichier des références
BIB_FILE = a.bib

# Dossier pour les fichiers temporaires
BUILD_DIR = build

# Commandes pour la compilation
LATEX = pdflatex
BIBER = biber

# Options de couleur pour les messages d'erreur et de succès
RED = \033[31m
GREEN = \033[32m
YELLOW = \033[33m
RESET = \033[0m

# Cibles par défaut
all: $(BUILD_DIR)/rapport.pdf

# Créer automatiquement le répertoire de build si nécessaire
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Règle pour compiler le fichier PDF
$(BUILD_DIR)/rapport.pdf: $(MAIN) $(BIB_FILE) | $(BUILD_DIR)
	@echo -e "$(YELLOW)Compilation de $(MAIN)...$(RESET)"
	$(LATEX) -interaction=nonstopmode -output-directory=$(BUILD_DIR) $(MAIN) || (echo -e "$(RED)Erreur dans la compilation de $(MAIN)$(RESET)" && exit 1)
	$(BIBER) --output-directory=$(BUILD_DIR) $(basename $(MAIN)) || (echo -e "$(RED)Erreur lors de l'exécution de Biber$(RESET)" && exit 1)
	$(LATEX) -interaction=nonstopmode -output-directory=$(BUILD_DIR) $(MAIN) || (echo -e "$(RED)Erreur dans la recompilation de $(MAIN)$(RESET)" && exit 1)
	$(LATEX) -interaction=nonstopmode -output-directory=$(BUILD_DIR) $(MAIN) || (echo -e "$(RED)Erreur dans la dernière compilation de $(MAIN)$(RESET)" && exit 1)
	@echo -e "$(GREEN)Compilation terminée avec succès.$(RESET)"
	@echo -e "$(YELLOW)Copie du PDF vers le répertoire principal...$(RESET)"
	rm -f rapport.pdf
	cp $(BUILD_DIR)/rapport.pdf .
	@echo -e "$(GREEN)Fichier PDF copié avec succès.$(RESET)"

# Règle pour compiler uniquement le texte et les images sans la bibliographie
tex:
	@echo -e "$(YELLOW)Compilation du texte et des images sans la bibliographie...$(RESET)"
	$(LATEX) -interaction=nonstopmode -output-directory=$(BUILD_DIR) $(MAIN) || (echo -e "$(RED)Erreur dans la compilation du texte$(RESET)" && exit 1)
	$(LATEX) -interaction=nonstopmode -output-directory=$(BUILD_DIR) $(MAIN) || (echo -e "$(RED)Erreur dans la recompilation du texte$(RESET)" && exit 1)
	@echo -e "$(GREEN)Compilation terminée avec succès.$(RESET)"
	@echo -e "$(YELLOW)Copie du PDF vers le répertoire principal...$(RESET)"
	rm -f rapport.pdf
	cp $(BUILD_DIR)/main.pdf rapport.pdf
	@echo -e "$(GREEN)Fichier PDF copié avec succès.$(RESET)"

# Règle pour ne compiler que la bibliographie si le fichier .bib a changé
bib:
	@echo -e "$(YELLOW)Exécution de Biber...$(RESET)"
	$(BIBER) --output-directory=$(BUILD_DIR) $(basename $(MAIN)) || (echo -e "$(RED)Erreur lors de l'exécution de Biber$(RESET)" && exit 1)

# Règle pour nettoyer les fichiers intermédiaires
clean:
	@echo -e "$(YELLOW)Suppression des fichiers temporaires...$(RESET)"
	rm -rf $(BUILD_DIR)/*.aux $(BUILD_DIR)/*.bbl $(BUILD_DIR)/*.blg $(BUILD_DIR)/*.log $(BUILD_DIR)/*.out $(BUILD_DIR)/*.toc $(BUILD_DIR)/*.lof $(BUILD_DIR)/*.lot $(BUILD_DIR)/*.fls $(BUILD_DIR)/*.fdb_latexmk $(BUILD_DIR)/*.pdf $(BUILD_DIR)/*.bcf $(BUILD_DIR)/*.xml
	rm -f rapport.pdf
	rm -rf $(BUILD_DIR)
	@echo -e "$(GREEN)Fichiers temporaires supprimés.$(RESET)"

# Cible pour recompiler complètement
rebuild: clean all

.PHONY: all clean rebuild bib tex