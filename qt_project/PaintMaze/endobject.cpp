#include <QGLWidget>
#include <QImage>
#include <GL/glu.h>
#include "endobject.h"
#include "collidable_a.h"
#include "mainglwidget.h"
#include "util.h"
#include "ball.h"

EndObject::EndObject(QVector3D *p) : Collidable_A(p)
{
    this->spacing = 2;
    this->height = 40;

    QImage image;
    image.load("res/tex/wall.png");
    image = QGLWidget::convertToGLFormat(image);

    this->image = image;

    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA , GL_UNSIGNED_BYTE, image.bits());
}

void EndObject::draw()
{
    glPushMatrix();
    createZoneEnd();
    createEndObject();
    glPopMatrix();
}

void EndObject::createTextEnd(QVector3D *pointPlayer)
{
    QVector3D *p1 = new QVector3D(*pointPlayer - *this->p);

    int distance = p1->length();
    QString strDistance = QString::number(distance)+"m";

    QFont mainTitle("Arial", 40);

    MainGLWidget::getInstance()->renderText(p->x()+(this->spacing/2)-1, this->p->y()+this->height+0.5, this->p->z()+(this->spacing/2)-1, strDistance , mainTitle);
    delete p1;
}

void EndObject::createEndObject()
{
    glEnable(GL_TEXTURE_2D);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    double textureEdge = this->spacing;
    glBindTexture (GL_TEXTURE_2D, texID);

    glColor4f(1, 1, 1, 1);

    glBegin(GL_QUADS);
    {
        // Top
        glTexCoord2d(0, 0);
        glVertex3f(p->x(), p->y()+this->height, p->z());
        glTexCoord2d(textureEdge, 0);
        glVertex3f(p->x(), p->y()+this->height, p->z()+this->spacing);
        glTexCoord2d(textureEdge, textureEdge);
        glVertex3f(p->x()+this->spacing, p->y()+this->height, p->z()+this->spacing);
        glTexCoord2d(0, textureEdge);
        glVertex3f(p->x()+this->spacing, p->y()+this->height, p->z());

        // Right
        glTexCoord2d(0, height);
        glVertex3f(p->x()+this->spacing, p->y()+this->height, p->z());
        glTexCoord2d(textureEdge, height);
        glVertex3f(p->x()+this->spacing, p->y()+this->height, p->z()+this->spacing);
        glTexCoord2d(textureEdge, 0);
        glVertex3f(p->x()+this->spacing, p->y(), p->z()+this->spacing);
        glTexCoord2d(0, 0);
        glVertex3f(p->x()+this->spacing, p->y(), p->z());


        // Front
        glTexCoord2d(textureEdge, height);
        glVertex3f(p->x()+this->spacing, p->y()+this->height, p->z()+this->spacing);
        glTexCoord2d(0, height);
        glVertex3f(p->x(), p->y()+this->height, p->z()+this->spacing);
        glTexCoord2d(0, 0);
        glVertex3f(p->x(), p->y(), p->z()+this->spacing);
        glTexCoord2d(textureEdge, 0);
        glVertex3f(p->x()+this->spacing, p->y(), p->z()+this->spacing);

        // Left
        glTexCoord2d(textureEdge, height);
        glVertex3f(p->x(), p->y()+this->height, p->z());
        glTexCoord2d(textureEdge, 0);
        glVertex3f(p->x(), p->y(), p->z());
        glTexCoord2d(0, 0);
        glVertex3f(p->x(), p->y(), p->z()+this->spacing);
        glTexCoord2d(0, height);
        glVertex3f(p->x(), p->y()+this->height, p->z()+this->spacing);


        // Behind
        glTexCoord2d(0, 0);
        glVertex3f(p->x(), p->y(), p->z());
        glTexCoord2d(0, this->height);
        glVertex3f(p->x(), p->y()+this->height, p->z());
        glTexCoord2d(textureEdge, this->height);
        glVertex3f(p->x()+this->spacing, p->y()+this->height, p->z());
        glTexCoord2d(textureEdge, 0);
        glVertex3f(p->x()+this->spacing, p->y(), p->z());

        // Bottom
        glTexCoord2d(0, 0);
        glVertex3f(p->x(), p->y(), p->z());
        glTexCoord2d(textureEdge, 0);
        glVertex3f(p->x()+this->spacing, p->y(), p->z());
        glTexCoord2d(textureEdge, textureEdge);
        glVertex3f(p->x()+this->spacing, p->y(), p->z()+this->spacing);
        glTexCoord2d(0, textureEdge);
        glVertex3f(p->x(), p->y(), p->z()+this->spacing);

    }
    glEnd();

    glDisable(GL_CULL_FACE);
    glDisable(GL_TEXTURE_2D);
}

void EndObject::createZoneEnd()
{
    glColor4f(1, 1, 1, 0.5);

    double zoneAroundEndObject = 5;

    glBegin(GL_QUADS);
    {
        // Bottom
        glVertex3f(p->x()-zoneAroundEndObject, p->y(), p->z()-zoneAroundEndObject);
        glVertex3f(p->x()+this->spacing+zoneAroundEndObject, p->y(), p->z()-zoneAroundEndObject);
        glVertex3f(p->x()+this->spacing+zoneAroundEndObject, p->y(), p->z()+this->spacing+zoneAroundEndObject);
        glVertex3f(p->x()-zoneAroundEndObject, p->y(), p->z()+this->spacing+zoneAroundEndObject);
    }
    glEnd();
}

bool EndObject::isColliding(Collidable_A *other)
{
    double rightBottomX = this->p->x()+this->spacing;
    double rightBottomZ = this->p->z()+this->spacing;

    double leftTopX = this->p->x();
    double leftTopZ = this->p->z();

    if(other->getP()->x() < rightBottomX && other->getP()->x() > leftTopX)
    {
        if(other->getP()->z() < rightBottomZ && other->getP()->z() > leftTopZ)
        {
            return true;
        }
    }
    return false;
}

void EndObject::createStain(QVector3D *p, QVector3D *, Ball *ball, double angle)
{

}
