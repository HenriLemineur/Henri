#include "Arme.h"
#include <iostream>

using namespace std;

Arme::Arme()
{
    m_nom ="tabouret";
    m_degats =20;
}

Arme::Arme(string nomArme, int degatsArme) : m_nom (nomArme), m_degats(degatsArme)
{

}

Arme::~Arme()
{

}

void Arme::changer(string nom, int degats)
{
    m_nom = nom;
    m_degats = degats;
}

int Arme::getdegat() const
{
    return m_degats;
}

string Arme::getnom() const
{
    return m_nom;
}

void Arme::afficher() const
{
    cout << "Arme : " << m_nom << " (Degats : " << m_degats << ")" << endl;
}