#include "Mage.h"
#include <iostream>

using namespace std;

Mage::Mage(): m_mana(50)
{

}

int Mage::getmana() const
{
    return m_mana;
}

void Mage::bouleDeFeu(Personnage& cible)
{
    int coutMana = 20; 
    if (m_mana >= coutMana) 
    {
        cible.recevoirDegats(40);
        m_mana -= coutMana;
        cout << "Le mage lance Boule de Feu !" << endl;
    }
    else
    {
        cout << "Mana insuffisant pour lancer Boule de Feu." << endl;
    }
}

void Mage::bouleDeGlace(Personnage& cible)
{
    int coutMana = 10; 
    if (m_mana >= coutMana) 
    {
        cible.recevoirDegats(30);
        m_mana -= coutMana;
        cout << "Le mage lance Boule de Glace !" << endl;
    }
    else
    {
        cout << "Mana insuffisant pour lancer Boule de Glace." << endl;
    }
}
void Mage::afficherEtat() const
{
    Personnage::afficherEtat();
    cout << "Mana : " << m_mana << endl;
}