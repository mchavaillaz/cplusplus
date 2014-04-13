#include <QGLWidget>
#include <QDebug>
#include <GL/glu.h>
#include <QKeyEvent>
#include <QTimer>
#include <QApplication>
#include "mainglwidget.h"
#include "scene_a.h"
#include "levelscene.h"
#include "camera.h"
#include "level1.h"
#include "player.h"
#include "keyboard.h"
#include "mouse.h"
#include "angle3d.h"
#include "mainmenuscene.h"
#include "loadingscene.h"

MainGLWidget* MainGLWidget::instance = 0;

MainGLWidget::MainGLWidget(QWidget *parent) : QGLWidget(parent)
{
    this->showFullScreen();
    this->setWindowTitle("PaintMaze");
    isProgramaticallySettingCursorPosition = true;
    this->isMainMenuMode = true;
    qApp->installEventFilter(this);
}

void MainGLWidget::init()
{
    MainMenuScene *mainMenuScene = new MainMenuScene();
    setCurrentScene(mainMenuScene);

    // main loop
    QTimer *timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(refresh()));
    timer->start(1000/60);
}

void MainGLWidget::enableMenuMode()
{
    this->isMainMenuMode = true;
    this->setCursor(Qt::ArrowCursor);
}

void MainGLWidget::enableLevelMode()
{
    LoadingScene *loadingScene = new LoadingScene();
    MainGLWidget::getInstance()->setCurrentScene(loadingScene);
    updateGL();

    this->isMainMenuMode = false;
    QVector3D p1(-5.0, 1.85, 5.0);
    Angle3D a1;
    Camera* camera = new Camera(&p1 , &a1);
    Level_A* level = new Level1();
    Player* player = new Player(new QVector3D(-5, 0, 5), camera, level, "player1");
    QPoint* point = new QPoint(width()/2, height()/2);
    LevelScene* levelScene = new LevelScene(camera, level, player, point);
    updateGL();
    this->setCursor(Qt::BlankCursor);
    setCurrentScene(levelScene);
}

bool MainGLWidget::eventFilter(QObject *, QEvent *event)
{
    if(event->type() == QEvent::MouseMove && isMainMenuMode)
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        Mouse::getInstance()->setMousePos(mouseEvent);
    }
    if (event->type() == QEvent::MouseMove && !isMainMenuMode)
    {
        if(isProgramaticallySettingCursorPosition)
        {
            QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
            Mouse::getInstance()->setMousePos(mouseEvent);
            isProgramaticallySettingCursorPosition = false;
            QCursor::setPos(width()/2, height()/2);
            isProgramaticallySettingCursorPosition = true;
        }
    }
    return false;
}

void MainGLWidget::refresh()
{
    updateGL();
}

MainGLWidget* MainGLWidget::getInstance()
{
    if(instance == 0)
    {
        instance = new MainGLWidget();
    }
    return instance;
}

void MainGLWidget::paintGL() {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glPushMatrix();
    this->currentScene->render();
    glPopMatrix();
}

void MainGLWidget::setCurrentScene(Scene_A* currentScene)
{
    this->currentScene = currentScene;
}

void MainGLWidget::initializeGL()
{
    glClearColor(0.2f, 0.7f, 1.0f, 1.0f);
    glEnable(GL_DEPTH);
    glEnable(GL_BLEND);
    glShadeModel(GL_SMOOTH);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void MainGLWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, this->width(), this->height());

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(30, (double)w/(double)h, 0.1, 10000);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void MainGLWidget::keyPressEvent(QKeyEvent* event)
{
    switch(event->key()) {
    case Qt::Key_Escape:
        exit(0);
        break;
    }
    Keyboard::getInstance()->addKey(event->key());
}

void MainGLWidget::keyReleaseEvent(QKeyEvent* event)
{
    Keyboard::getInstance()->removeKey(event->key());
}

void MainGLWidget::mouseMoveEvent(QMouseEvent*)
{
    // rien
}

void MainGLWidget::mousePressEvent(QMouseEvent *event)
{
    Mouse::getInstance()->setButtonsDown(event);
}

void MainGLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    Mouse::getInstance()->setButtonsUp(event);
}

void MainGLWidget::wheelEvent(QWheelEvent *event)
{
    Mouse::getInstance()->setScroll(event);
}
