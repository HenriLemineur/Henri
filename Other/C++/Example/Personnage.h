#ifndef DEF_PERSONNAGE
#define DEF_PERSONNAGE

#include <string>

#include <iostream>
#include "Arme.h"
using namespace std;

class Personnage
{
public:

    Personnage();
    Personnage(int vie, string nom_arme, int degat_arme);
    void recevoirDegats(int nbDegats);
    void attaquer(Personnage& cible);
    void boirePotionDeVie(int quantitePotion);
    bool estVivant();
    void afficherEtat() const;
    int getVie() const;
    ~Personnage();
protected:

    int m_vie;
    Arme* m_arme;
};
#endif