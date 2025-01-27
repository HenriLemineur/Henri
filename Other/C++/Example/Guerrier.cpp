#include "Guerrier.h"
#include <iostream>

Guerrier::Guerrier(int vie, std::string nom_arme, int degat_arme, int pointsArmure)
    : Personnage(vie, nom_arme, degat_arme), m_armure(new Armure(pointsArmure))
{
}


Guerrier::~Guerrier()
{
    delete m_armure;
}

void Guerrier::recevoirDegats(int nbDegats)
{
    
    if (m_armure && m_armure->getPointsArmure() > 0)
    {
        m_armure->reduirePointsArmure(nbDegats);
        nbDegats = (m_armure->getPointsArmure() < 0) ? -m_armure->getPointsArmure() : 0; 
    }

    
    if (nbDegats > 0)
    {
        m_vie -= nbDegats;
        if (m_vie < 0)
        {
            m_vie = 0;
        }
    }
}


void Guerrier::afficherEtat() const
{
    
    Personnage::afficherEtat();

    if (m_armure) {
        std::cout << "Points d'armure : " << m_armure->getPointsArmure() << std::endl;
    }
    else {
        std::cout << "Pas d'armure équipée." << std::endl;
    }
}

int Guerrier::getArmurePoints() const
{
    return m_armure ? m_armure->getPointsArmure() : 0;
}

void Guerrier::equiperArmure(int pointsArmure)
{
    if (m_armure) {
        delete m_armure;  
    }
    m_armure = new Armure(pointsArmure);  
}
