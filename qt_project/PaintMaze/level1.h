#ifndef LEVEL1_H
#define LEVEL1_H

#include "level_a.h"

class EndObject;

class Level1 : public Level_A
{
public:
    Level1();

private:
    void fillListFormNonCollidable();
    void fillListFormCollidable();
};

#endif // LEVEL1_H
