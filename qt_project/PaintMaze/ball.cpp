#include <QGLWidget>
#include <GL/glu.h>
#include <cmath>
#include "ball.h"

Ball::Ball(QVector3D *p, Angle3D *angle, QColor color, double vitesse) : Movable(p, angle, vitesse)
{
    this->color = color;
}

bool Ball::isColliding(Collidable_A *a)
{
    return a->isColliding(this);
}

// http://www.opengl.org/discussion_boards/showthread.php/163561-How-to-posistion-a-gluSphere
void Ball::draw()
{
    double rayon = 0.05;

    glPushMatrix();
    glTranslatef(p->x(), p->y(), p->z());

    glLineWidth( 3.0 );
    glColor3f(this->color.red()/255.0, this->color.green()/255.0, this->color.blue()/255.0);

    glBegin( GL_LINE_LOOP );
    GLUquadricObj *quadric;
    quadric = gluNewQuadric();

    gluQuadricDrawStyle(quadric, GLU_FILL );
    gluSphere( quadric , rayon , 36 , 18 );

    gluDeleteQuadric(quadric);
    glEndList();

    glEnd();
    glPopMatrix();
}

void Ball::createStain(QVector3D *p, QVector3D *a, Ball *ball, double angle)
{

}
