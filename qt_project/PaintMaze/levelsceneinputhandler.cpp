#include <QCursor>
#include <QVector3D>
#include "levelsceneinputhandler.h"
#include "keyboard.h"
#include "player.h"
#include "mouse.h"
#include "level_a.h"
#include "angle3d.h"
#include "camera.h"
#include "wall.h"
#include "endobject.h"
#include "mainglwidget.h"

LevelSceneInputHandler::LevelSceneInputHandler(Player* player, QPoint *point, Level_A *level, Camera *camera)
{
    this->keyboard = Keyboard::getInstance();
    this->player = player;
    this->mouse = Mouse::getInstance();
    this->level = level;

    this->cameraCopy = new Camera(new QVector3D(0,0,0), new Angle3D(0,0,0));
    this->camera = camera;
    this->playerCopy = new Player(player->getP(), cameraCopy, level, QString("playerCopy"));
    this->playerCopyX = new Player(player->getP(), cameraCopy, level, QString("playerCopyX"));
    this->playerCopyZ = new Player(player->getP(), cameraCopy, level, QString("playerCopyZ"));
    this->listCollidableCopy = level->getListShapeCollisionnable();
}

void LevelSceneInputHandler::handleInputs()
{
    delete playerCopy;
    delete cameraCopy;
    this->cameraCopy = new Camera(camera->getPoint(), camera->getAngle());
    this->playerCopy = new Player(player->getP(), cameraCopy, level, QString("playerCopy"));

    player->actionLook(mouse->getDx(), mouse->getDy());

    if(keyboard->isKeyDown(Qt::Key_F) || mouse->isLeftButtonDown())
    {
            player->actionTirer();
    }

    if(mouse->getScrollDy() > 0)
    {
        player->nextColor();
        mouse->setScrollDy(0);
    }
    else if(mouse->getScrollDy() < 0)
    {
        player->previousColor();
        mouse->setScrollDy(0);
    }

    if(keyboard->isKeyPairDown(Qt::Key_W, Qt::Key_D))
    {
        playerCopy->actionDiagUpRight();
        if(!isColliding(playerCopy))
        {
            player->actionDiagUpRight();
        }
        return;
    }

    else if(keyboard->isKeyPairDown(Qt::Key_W, Qt::Key_A))
    {
        playerCopy->actionDiagUpLeft();
        if(!isColliding(playerCopy))
        {
            player->actionDiagUpLeft();
        }
        return;
    }

    else if(keyboard->isKeyPairDown(Qt::Key_S, Qt::Key_A))
    {
        playerCopy->actionDiagDownLeft();
        if(!isColliding(playerCopy))
        {
            player->actionDiagDownLeft();
        }
        return;
    }

    else if(keyboard->isKeyPairDown(Qt::Key_S, Qt::Key_D))
    {
        playerCopy->actionDiagDownRight();
        if(!isColliding(playerCopy))
        {
            player->actionDiagDownRight();
        }
        return;
    }

    if(mouse->isLeftButtonDown())
    {
        //player->actionTirer();
    }

    if(keyboard->isKeyDown(Qt::Key_W))
    {
        playerCopy->actionGoStraight();
        if(!isColliding(playerCopy))
        {
            player->actionGoStraight();
        }
    }

    if(keyboard->isKeyDown(Qt::Key_S))
    {
        playerCopy->actionGoBackwards();
        if(!isColliding(playerCopy))
        {
            player->actionGoBackwards();
        }
    }

    if(keyboard->isKeyDown(Qt::Key_A))
    {
        playerCopy->actionStrafeLeft();
        if(!isColliding(playerCopy))
        {
            player->actionStrafeLeft();
        }
    }

    if(keyboard->isKeyDown(Qt::Key_D))
    {
        playerCopy->actionStrafeRight();
        if(!isColliding(playerCopy))
        {
            player->actionStrafeRight();
        }
    }
}

bool LevelSceneInputHandler::isColliding(Player *playerCopy)
{
    QFont font("Times", 25, QFont::Bold);
    glColor3f(0.0, 0.0, 0.0);

    bool isPlayerCollidingX = false;
    bool isPlayerCollidingZ = false;

    QVector3D *pcx = new QVector3D(this->playerCopy->getP()->x(), 0, this->player->getP()->z());
    QVector3D *pcz = new QVector3D(this->player->getP()->x(), 0, this->playerCopy->getP()->z());

    playerCopyX->setP(pcx);
    playerCopyZ->setP(pcz);

    foreach(Collidable_A *element, listCollidableCopy)
    {
        if(element->isColliding(playerCopyX))
        {
//            MainGLWidget::getInstance()->renderText(10, 130, "playerCopyX colliding X:" + QString::number(element->getP()->getX()) + " Z:" + QString::number(element->getP()->getZ()), font);
            isPlayerCollidingX = true;

            foreach(Collidable_A *element2, listCollidableCopy)
            {
                if(element2->isColliding(playerCopyZ))
                {
                isPlayerCollidingZ = true;
//                MainGLWidget::getInstance()->renderText(10, 160, "playerCopyZ colliding X:" + QString::number(element2->getP()->getX()) + " Z:" + QString::number(element2->getP()->getZ()), font);
                break;
                }
            }//End foreach element2 in listCollidableCopy

            break;
        }
        else if(element->isColliding(playerCopyZ))
        {
//            MainGLWidget::getInstance()->renderText(10, 160, "playerCopyZ colliding Z:" + QString::number(element->getP()->getX()) + " Z:" + QString::number(element->getP()->getZ()), font);
            isPlayerCollidingZ = true;

            foreach(Collidable_A *element2, listCollidableCopy)
            {
                if(element2->isColliding(playerCopyX))
                {
                isPlayerCollidingX = true;
//                MainGLWidget::getInstance()->renderText(10, 130, "playerCopyX colliding X:" + QString::number(element2->getP()->getX()) + " Z:" + QString::number(element2->getP()->getZ()), font);
                break;
                }
            }//End foreach element2 in listCollidableCopy

            break;
        }
    }//End foreach element in listCollidableCopy

    bool isPlayerCollidableEndObject = this->level->getEndObject()->isColliding(playerCopyZ);

    if(isPlayerCollidingX && isPlayerCollidingZ)
    {
//        MainGLWidget::getInstance()->renderText(10, 100, "CORNER", font);

        delete pcx;
        delete pcz;

        return true;
    }
    else if(isPlayerCollidingX)
    {
//        MainGLWidget::getInstance()->renderText(10, 100, "slideZ", font);

        player->slide(this->playerCopyZ->getP()->x(), this->playerCopyZ->getP()->z());

        delete pcx;
        delete pcz;

        return true;
    }
    else if(isPlayerCollidingZ)
    {
//        MainGLWidget::getInstance()->renderText(10, 100, "slideX", font);

        player->slide(this->playerCopyX->getP()->x(),this->playerCopyX->getP()->z());

        delete pcx;
        delete pcz;

        return true;
    }
    else if(isPlayerCollidableEndObject)
    {
        return true;
    }
    else
    {
        delete pcx;
        delete pcz;

        return false;
    }
}
