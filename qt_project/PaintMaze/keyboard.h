#ifndef KEYBOARD_H
#define KEYBOARD_H

#include <QKeyEvent>
#include <QLinkedList>

class Keyboard
    {
    public:
        static Keyboard* getInstance();
        bool isKeyDown(Qt::Key key);
        bool isKeyDown(int key);
        bool isKeyPairDown(Qt::Key, Qt::Key);
        bool isKeyPairDown(int, int);
        void addKey(Qt::Key key);
        void addKey(int key);
        bool removeKey(Qt::Key key);
        bool removeKey(int key);

    private :
        Keyboard();
        QLinkedList<Qt::Key> listKeysPressed;

        static Keyboard* instance;
    };

#endif // KEYBOARD_H
