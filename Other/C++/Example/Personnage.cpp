#include "Personnage.h"
#include <iostream>
#include "Arme.h"
using namespace std;

Personnage::Personnage() : m_arme(0)
{
    m_vie = 100;
    m_arme =new Arme();
    
}

Personnage::Personnage(int vie, string nom_arme, int degat_arme) : m_vie(vie), m_arme(0)

{
    m_arme = new Arme(nom_arme, degat_arme);
}


Personnage::~Personnage()
{
    delete m_arme;
}


void Personnage::recevoirDegats(int nbDegats)
{
    m_vie -= nbDegats;

    if (m_vie < 0)
    {
        m_vie = 0; 
    }
}

void Personnage::attaquer(Personnage & cible)
{
    cible.recevoirDegats(m_arme->getdegat());
   
}

void Personnage::boirePotionDeVie(int quantitePotion)
{
    m_vie += quantitePotion;

    if (m_vie > 100)
    {
        m_vie = 100;
    }
}

bool Personnage::estVivant()
{
    return m_vie > 0;
}

void Personnage::afficherEtat() const
{
    cout << "Vie : " << m_vie << endl;
    cout << "Arme : " << m_arme->getnom() << endl;
    cout << "Degats : " << m_arme->getdegat() << endl;
    
    
}
int Personnage::getVie() const
{
    return m_vie;
}