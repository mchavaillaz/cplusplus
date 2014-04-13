#ifndef LEVELSCENE_H
#define LEVELSCENE_H

#include <QVector>
#include <QList>
#include <QtAlgorithms>
#include <QGLShaderProgram>
#include "scene_a.h"
#include "level_a.h"

class Widget;
class Camera;
class Level_A;
class Player;
class Ground;
class Landscape;
class LevelSceneInputHandler;
class Stain;

class LevelScene : public Scene_A
{
    Q_OBJECT
public:
    LevelScene(Camera* camera, Level_A* level, Player* player, QPoint *point);
    ~LevelScene();
    void render();
    Camera* getCamera();
    Player* getPlayer();

public slots:
    void slot_addTache(Stain *);

private:
    Camera* camera;
    Level_A* level;
    Player* player;
    Ground* ground;
    Landscape *landscape;
    LevelSceneInputHandler* levelSceneInputHandler;
    QVector<Stain*> listStain;
    QList<Widget*> listWidget;
    QGLShaderProgram *stainShaderProgram;
    int i;

};

#endif // LEVELSCENE_H
