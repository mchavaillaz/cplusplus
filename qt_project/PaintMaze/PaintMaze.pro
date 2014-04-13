
#-------------------------------------------------
#
# Project created by QtCreator 2012-11-02T15:52:16
#
#-------------------------------------------------
QT += core gui
QT += opengl
CONFIG += rtti
# pour que ca compile sous linux
#LIBS += -lGLU
TARGET = PaintMaze
TEMPLATE = app
SOURCES += main.cpp\
    scene_a.cpp \
    inputhandler_a.cpp \
    level_a.cpp \
    collidable_a.cpp \
    levelscene.cpp \
    camera.cpp \
    player.cpp \
    widget.cpp \
    levelsceneinputhandler.cpp \
    mainmenuinputhandler.cpp \
    mainglwidget.cpp \
    level1.cpp \
    keyboard.cpp \
    mouse.cpp \
    ground.cpp \
    angle3d.cpp \
    wall.cpp \
    stain.cpp \
    face.cpp \
    util.cpp \
    shape_a.cpp \
    movable.cpp \
    ball.cpp \
    mainmenuscene.cpp \
    landscape.cpp \
    endobject.cpp \
    ammowidget.cpp \
    timewidget.cpp \
    timeupscene.cpp \
    timeupmenuinputhandler.cpp \
    loadingscene.cpp \
    ballcolorwidget.cpp \
    levelfinishedscene.cpp \
    levelfinishedinputhandler.cpp \
    creditsscene.cpp \
    creditsinputhandler.cpp \
    qray3d.cpp \
    qplane3d.cpp

HEADERS  += paintmazewindow.h \
    scene_a.h \
    inputhandler_a.h \
    level_a.h \
    shape_a.h \
    collidable_a.h \
    levelscene.h \
    camera.h \
    player.h \
    widget.h \
    levelsceneinputhandler.h \
    mainmenuinputhandler.h \
    mainglwidget.h \
    level1.h \
    keyboard.h \
    mouse.h \
    ground.h \
    angle3d.h \
    wall.h \
    stain.h \
    face.h \
    util.h \
    movable.h \
    ball.h \
    mainmenuscene.h \
    landscape.h \
    endobject.h \
    ammowidget.h \
    timewidget.h \
    timeupscene.h \
    timeupmenuinputhandler.h \
    loadingscene.h \
    ballcolorwidget.h \
    levelfinishedscene.h \
    levelfinishedinputhandler.h \
    creditsscene.h \
    creditsinputhandler.h \
    qray3d.h \
    qplane3d.h

FORMS    += paintmazewindow.ui
OTHER_FILES += \
    Stain.frag \
    Stain.vert

