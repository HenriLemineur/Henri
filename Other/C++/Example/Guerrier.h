#ifndef DEF_GUERRIER
#define DEF_GUERRIER

#include "Personnage.h"
#include "Armure.h"

class Guerrier : public Personnage
{
public:
    Guerrier(int vie, std::string nom_arme, int degat_arme, int pointsArmure);
    ~Guerrier();  

    void recevoirDegats(int nbDegats);
    void afficherEtat() const;

    int getArmurePoints() const;
    void equiperArmure(int pointsArmure);

private:
    Armure* m_armure;  
};

#endif
