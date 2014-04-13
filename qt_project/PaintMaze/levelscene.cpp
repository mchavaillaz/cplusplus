#include <QFile>
#include <QDebug>
#include "levelscene.h"
#include "mainglwidget.h"
#include "camera.h"
#include "player.h"
#include "ground.h"
#include "stain.h"
#include "landscape.h"
#include "movable.h"
#include "levelsceneinputhandler.h"
#include "collidable_a.h"
#include "widget.h"
#include "endobject.h"
#include "mouse.h"
#include "ammowidget.h"
#include "timewidget.h"
#include "timeupscene.h"
#include "ballcolorwidget.h"
#include "ball.h"
#include "levelfinishedscene.h"
#include "face.h"

LevelScene::LevelScene(Camera* camera, Level_A* level, Player* player, QPoint *point)
{
    this->camera = camera;
    this->level = level;
    this->player = player;
    this->ground = new Ground();
    this->landscape = new Landscape();
    this->levelSceneInputHandler = new LevelSceneInputHandler(player, point, level, camera);
    QVector3D *tmp = new QVector3D(*level->getStartingPoint());
    this->player->setP(tmp);
    delete tmp;
    this->camera->setP(level->getStartingPoint());

    this->stainShaderProgram = new QGLShaderProgram(this);
    this->stainShaderProgram->addShaderFromSourceFile(QGLShader::Fragment, "shaders/Stain.frag");
    //this->stainShaderProgram->addShaderFromSourceFile(QGLShader::Vertex, "../PaintMaze/Stain.vert");
    this->stainShaderProgram->link();
    this->i = 1;

    listWidget.append(new AmmoWidget(this->player));

    TimeWidget *timeWidget = new TimeWidget(this->level);
    listWidget.append(timeWidget);

    listWidget.append(new BallColorWidget(this->player));


    connect(ground, SIGNAL(signal_addTache(Stain*)), this, SLOT(slot_addTache(Stain*)));
    foreach(Collidable_A* element, this->level->getListShapeCollisionnable())
    {
        connect(element, SIGNAL(signal_addTache(Stain*)), this, SLOT(slot_addTache(Stain*)));
    }

    level->start();
}

Camera* LevelScene::getCamera() {
    return this->camera;
}

Player* LevelScene::getPlayer() {
    return this->player;
}

void LevelScene::render()
{
    //Time up
    if(this->level->getTimeLeftMillis() <= 0)
    {
        MainGLWidget::getInstance()->enableMenuMode();
        MainGLWidget::getInstance()->setCurrentScene(new TimeUpScene());
    }


    // Reached goal
    if(this->level->getEndObject()->isColliding(this->player)) {
        MainGLWidget::getInstance()->enableMenuMode();
        MainGLWidget::getInstance()->setCurrentScene(new LevelFinishedScene(this->level->getTimeLeftMillis(), this->player->getAmmo()));
    }

    camera->view();
    ground->draw();
    landscape->draw();

    glPushMatrix();
    glScaled(-1, 1, -1);

    this->levelSceneInputHandler->handleInputs();

    // dessin des murs
    glColor4d(1, 1, 1, 0);
//    foreach(Collidable_A* element, this->level->getListShapeCollisionnable())
//    {
//        element->draw();
//    }

    this->level->getEndObject()->draw();
    this->level->getEndObject()->createTextEnd(this->player->getP());

    // dessin des taches
    this->stainShaderProgram->bind();

    foreach(Stain* element, this->listStain)
    {
        QColor _color = element->getColor();
        if(element->getFace()->getNormal()->x() != 0 )
            {
            _color = _color.lighter(120);
            }
        else if(element->getFace()->getNormal()->z() != 0)
            {
            _color = _color.darker(120);
            }
        QVector4D color(_color.redF(), _color.greenF(), _color.blueF(), 1);
        this->stainShaderProgram->setUniformValue("color", color);
        element->draw();
    }
    this->stainShaderProgram->release();

    QList<Movable *> listeAEffacer;
    foreach(Movable* element, this->level->getListMovable())
    {
        element->moveTick();
        element->draw();
        foreach(Collidable_A* e, this->level->getListShapeCollisionnable())
        {
            if(e->isColliding(element) || ground->isColliding(element))
            {
                if(ground->isColliding(element))
                    ground->createStain(new QVector3D(*element->getP()), element->getPointOrigine(), (Ball*)element);
                else
                    e->createStain(new QVector3D(*element->getP()),element->getPointOrigine(), (Ball*)element);
                listeAEffacer.append(element);
                break;
            }
        }
    }
    foreach(Movable* element, listeAEffacer)
    {
        this->level->removeAndDeleteMovable(element);
    }

    glPopMatrix();

    foreach(Widget* element, this->listWidget)
    {
        element->draw();
    }

    glPopMatrix();

    glColor3d(1, 1, 1);
}

void LevelScene::slot_addTache(Stain *stain)
{
    listStain.push_back(stain);
}

LevelScene::~LevelScene()
{
    try
    {
        qDeleteAll(listStain.begin(), listStain.end());
        qDeleteAll(listWidget.begin(), listWidget.end());
    }
    catch(...)
    {
        qDebug() << "destruction de levelscene erreur";
    }
}
