#ifndef MAINGLWIDGET_H
#define MAINGLWIDGET_H

#include <QGLWidget>
#include <QMutex>

class Scene_A;

class MainGLWidget : public QGLWidget
{
    Q_OBJECT
public:
    static MainGLWidget* getInstance();
    void setCurrentScene(Scene_A* currentScene);
    void enableLevelMode();
    void enableMenuMode();

    void paintGL();
    void resizeGL(int w, int h);
    void initializeGL();

    void keyPressEvent(QKeyEvent *);
    void keyReleaseEvent(QKeyEvent *);
    bool eventFilter(QObject *, QEvent *event);
    void mouseMoveEvent(QMouseEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void wheelEvent(QWheelEvent *);

    void init();

private slots:
    void refresh();

private:
    MainGLWidget(QWidget *parent = 0);

    Scene_A* currentScene;
    static MainGLWidget* instance;
    bool isProgramaticallySettingCursorPosition;
    bool isMainMenuMode;

signals:

public slots:

};

#endif // MAINGLWIDGET_H
