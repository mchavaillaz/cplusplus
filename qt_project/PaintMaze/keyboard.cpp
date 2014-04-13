#include "keyboard.h"

Keyboard* Keyboard::instance = 0;

Keyboard* Keyboard::getInstance()
    {
    if(Keyboard::instance == 0)
        {
        Keyboard::instance = new Keyboard();
        }

    return Keyboard::instance;
    }

bool Keyboard::isKeyDown(Qt::Key key)
    {
    return this->listKeysPressed.contains(key);
    }

bool Keyboard::isKeyDown(int key)
    {
    return this->isKeyDown((Qt::Key)key);
    }

bool Keyboard::isKeyPairDown(Qt::Key key1, Qt::Key key2)
    {
    return this->isKeyDown(key1) and this->isKeyDown(key2);
    }

bool Keyboard::isKeyPairDown(int key1, int key2)
    {
    return this->isKeyPairDown((Qt::Key)key1, (Qt::Key)key2);
    }

void Keyboard::addKey(Qt::Key key)
    {
    this->listKeysPressed.append(key);
    }

void Keyboard::addKey(int key)
    {
    this->addKey((Qt::Key)key);
    }

bool Keyboard::removeKey(Qt::Key key)
    {
    return this->listKeysPressed.removeOne(key);
    }

bool Keyboard::removeKey(int key)
    {
    return this->removeKey((Qt::Key)key);
    }

Keyboard::Keyboard()
    {
    }
