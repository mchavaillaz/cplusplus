#ifndef LOADINGSCENE_H
#define LOADINGSCENE_H

#include "scene_a.h"

class LoadingScene : public Scene_A
{
public:
    LoadingScene();
    void render();
private:
    QFont *font;
};

#endif // LOADINGSCENE_H
