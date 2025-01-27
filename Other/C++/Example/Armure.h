#ifndef DEF_ARMURE
#define DEF_ARMURE


#include <iostream>
#include <string>

class Armure
{
public:
    Armure(int pointsArmure);
    int getPointsArmure() const;
    void reduirePointsArmure(int degats);
private:
    int m_pointsArmure;  
};

#endif
