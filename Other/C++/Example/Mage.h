#ifndef DEF_MAGE
#define DEF_MAGE

#include <iostream>
#include <string>
#include "Personnage.h"

class Mage : public Personnage
{
public:
    Mage();
    void bouleDeFeu(Personnage& cible);
    void bouleDeGlace(Personnage& cible);
    int getmana() const;
    void afficherEtat() const;

private:
    int m_mana;
};

#endif