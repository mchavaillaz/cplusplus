#ifndef SCENE_A_H
#define SCENE_A_H

#include <QGLWidget>
class Camera;

class Scene_A : public QObject
{
    Q_OBJECT
public:
    Scene_A();
    virtual ~Scene_A() {}
    virtual void render() = 0;

protected:
    Camera *camera;
};

#endif // SCENE_A_H
