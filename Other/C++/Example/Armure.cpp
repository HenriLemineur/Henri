#include "Armure.h"

Armure::Armure(int pointsArmure) : m_pointsArmure(pointsArmure)
{
}

int Armure::getPointsArmure() const
{
    return m_pointsArmure;
}

void Armure::reduirePointsArmure(int degats)
{
    if (m_pointsArmure > 0)
    {
        m_pointsArmure -= degats;
        if (m_pointsArmure < 0)
        {
            m_pointsArmure = 0;  
        }
    }
}