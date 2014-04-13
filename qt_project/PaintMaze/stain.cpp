#include <QGLWidget>
#include <cmath>
#include <QGLBuffer>
#include <QVector3D>
#include "util.h"
#include "stain.h"
#include "face.h"
Stain::Stain(Face *face, QVector3D *point, QVector3D* pointOrigine , QColor *color) : Shape_A(point)
{
    this->face = face;
    this->pOrigine = pointOrigine;
    this->pModif = new QVector3D(this->p->x(), this->p->y(), this->p->z());
    this->color = *color;
    delete color;
//    this->face->projectPoint(this->pModif);
    this->taille = 300;
    GLfloat *vertices = new GLfloat[taille];
    GLuint indices[taille];
    QVector3D *tmp = new QVector3D(this->pModif->x(),this->pModif->y(),this->pModif->z());
    double distance = 0;
    this->face->calculDistance(tmp, &distance);
    this->face->projectPoint(tmp);
    indices[0] = 0;
    vertices[0] = tmp->x();
    vertices[1] = tmp->y();
    vertices[2] = tmp->z();
    delete tmp;
    int elementEnleve = 0;
    int cpt = 1;
    for(int i = 3; i < taille; i+=3)
    {
        QVector3D *tmp = new QVector3D(this->pModif->x(),this->pModif->y(),this->pModif->z());
        tmp->setX(tmp->x() + getRandomRayon()*cos(getRandomAlpha())*cos(getRandomBeta()));
        tmp->setY(tmp->y() + getRandomRayon()*cos(getRandomAlpha())*sin(getRandomBeta()));
        tmp->setZ(tmp->z() + getRandomRayon()*sin(getRandomAlpha()));


        this->face->projectPoint(tmp);

        if(Util::isPointInSideFace(tmp, this->face))
        {
            vertices[i] = tmp->x();
            vertices[i+1] = tmp->y();
            vertices[i+2] = tmp->z();
            indices[cpt] = cpt;
            cpt ++;
        }
        else
        {
           elementEnleve+=3;
           i = i - 3;
           //cpt --;
        }
        delete tmp;
    }
    // upload sur gpu
    this->buffer = new QGLBuffer(QGLBuffer::VertexBuffer);
    this->buffer->create();
    this->buffer->bind();
    this->buffer->allocate(vertices, (taille) * sizeof(GLfloat));
    this->buffer->release();
    this->buffer2 = new QGLBuffer(QGLBuffer::IndexBuffer);
    this->buffer2->create();
    this->buffer2->bind();
    this->buffer2->allocate(indices, (cpt) * sizeof(GLuint));
    this->buffer2->release();
    delete vertices;
    //delete indices;
}
void Stain::draw()
{
    glColor3f(color.red()/255.0,color.green()/255.0, color.blue()/255.0);
    glEnableClientState(GL_VERTEX_ARRAY);
    this->buffer->bind();
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    this->buffer->release();
    this->buffer2->bind();
    glDrawElements(GL_POLYGON, this->taille, GL_UNSIGNED_INT, NULL);
    this->buffer2->release();
    glDisableClientState(GL_VERTEX_ARRAY);
}
double Stain::getRandomRayon()
{
    return qrand() % 2 + 1.5;
}
double Stain::getRandomAlpha()
{
    int degree = (qrand() % 181) - 90;
    return ((double)degree / (double)180.0) * (double)M_PI;
}
double Stain::getRandomBeta()
{
    int degree = (qrand() % 361) - 180;
    return ((double)degree / (double)180.0) * (double)M_PI;
}
