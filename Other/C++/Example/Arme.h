#ifndef DEF_ARME
#define DEF_ARME

#include <iostream>
#include <string>

using namespace std;

class Arme
{
public:

    Arme();
    Arme(string nom, int degats);
    void changer(std::string nom, int degats);
    void afficher() const;
    int getdegat() const;
    string getnom() const;
    ~Arme();
private:

    string m_nom;
    int m_degats;
};

#endif