#include <iostream>
#include "Personnage.h"
#include "Mage.h"
#include "Guerrier.h"
#include "Arme.h"
#include "Armure.h"

using namespace std;

int main()
{
    
    Guerrier guerrier1(100, "baton", 15, 50);
    cout << "État initial du Guerrier:" << endl;
    guerrier1.afficherEtat();
    cout << endl;

    Mage mage1;
    cout << "État initial du Mage:" << endl;
    mage1.afficherEtat();
    cout << endl;


    cout << "Le Mage attaque le Guerrier:" << endl;
    mage1.attaquer(guerrier1);
    guerrier1.afficherEtat();  
    mage1.afficherEtat();      
    cout << endl;


    cout << "Le Guerrier attaque le Mage:" << endl;
    guerrier1.attaquer(mage1);
    mage1.afficherEtat();     
    cout << endl;

 
    cout << "Le Guerrier attaque encore pour épuiser l'armure:" << endl;
    guerrier1.recevoirDegats(30);  
    guerrier1.afficherEtat();     
    cout << endl;

  
    cout << "Le Guerrier subit encore des dégâts après l'épuisement de l'armure:" << endl;
    guerrier1.recevoirDegats(30); 
    guerrier1.afficherEtat();     
    cout << endl;

    cout << "Le Guerrier subit encore des dégâts après l'épuisement de l'armure:" << endl;
    guerrier1.recevoirDegats(30);  
    guerrier1.afficherEtat();      
    cout << endl;

    return 0;
}
