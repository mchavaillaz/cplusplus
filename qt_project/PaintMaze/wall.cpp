#include <QGLWidget>
#include <QGLBuffer>
#include <QVector3D>
#include <QDebug>
#include "plane3d.h"
#include "ray3d.h"
#include "wall.h"
#include "player.h"
#include "mainglwidget.h"
#include "face.h"
#include "stain.h"
#include "ball.h"

double Wall::height = 20;

Wall::Wall(QVector3D *p1, QVector3D *p2, QVector3D *p3, QVector3D *p4) : Collidable_A(p1)
    {
    this->p2 = new QVector3D(p2->x(),p2->y(),p2->z());
    this->p3 = new QVector3D(p3->x(),p3->y(),p3->z());
    this->p4 = new QVector3D(p4->x(),p4->y(),p4->z());

    addFace(this->p, this->p2, new QVector3D(this->p2->x(), this->height, this->p2->z()), new QVector3D(this->p->x(), this->height,this->p->z()));
    addFace(this->p2, this->p3, new QVector3D(this->p3->x(), this->height, this->p3->z()) , new QVector3D(this->p2->x(), this->height, this->p2->z()));
    addFace(this->p3, this->p4, new QVector3D(p4->x(), this->height, p4->z()) , new QVector3D(p3->x(), this->height, this->p3->z()));
    addFace(this->p4, this->p, new QVector3D(this->p->x(), this->height, this->p->z()) , new QVector3D(this->p4->x(), this->height, this->p4->z()));
    addFace(new QVector3D(this->p->x(), this->height, this->p->z()), new QVector3D(this->p2->x(), this->height, this->p2->z()), new QVector3D(this->p3->x(), this->height, this->p3->z()), new QVector3D(this->p4->x(), this->height, this->p4->z()));
    addFace(this->p2, this->p, this->p4, this->p3);

    const int taille = 24;

    GLfloat *vertices = new GLfloat[taille];
    GLuint indices[taille] =
    {
        0,1,5,4,
        1,2,6,5,
        2,3,7,6,
        3,0,4,7,
        4,5,6,7,
        1,0,3,2
    };

    // vertices 0
    vertices[0] = p->x();
    vertices[1] = p->y();
    vertices[2] = p->z();

    // vertices 1
    vertices[3] = this->p2->x();
    vertices[4] = this->p2->y();
    vertices[5] = this->p2->z();

    // vertices 2
    vertices[6] = this->p3->x();
    vertices[7] = this->p3->y();
    vertices[8] = this->p3->z();

    // vertices 3
    vertices[9] = this->p4->x();
    vertices[10] = this->p4->y();
    vertices[11] = this->p4->z();

    // vertices 4
    vertices[12] = p->x();
    vertices[13] = this->height;
    vertices[14] = p->z();

    // vertices 5
    vertices[15] = this->p2->x();
    vertices[16] = this->height;
    vertices[17] = this->p2->z();

    // vertices 6
    vertices[18] = this->p3->x();
    vertices[19] = this->height;
    vertices[20] = this->p3->z();

    // vertices 7
    vertices[21] = this->p4->x();
    vertices[22] = this->height;
    vertices[23] = this->p4->z();

    // upload sur gpu
    this->buffer = new QGLBuffer(QGLBuffer::VertexBuffer);
    this->buffer->create();
    this->buffer->bind();
    this->buffer->allocate(vertices, taille * sizeof(GLfloat));
    this->buffer->release();

    this->buffer2 = new QGLBuffer(QGLBuffer::IndexBuffer);
    this->buffer2->create();
    this->buffer2->bind();
    this->buffer2->allocate(indices, taille * sizeof(GLuint));
    this->buffer2->release();

    delete vertices;
    //delete indices;
    }

void Wall::addFace(QVector3D *p1, QVector3D *p2, QVector3D *p3, QVector3D *p4)
    {
    Face *tmp = new Face();
    tmp->add(p1);
    tmp->add(p2);
    tmp->add(p3);
    tmp->add(p4);

    listFace.push_back(tmp);
    }

void Wall::createStain(QVector3D *p, QVector3D* pOrigine, Ball *ball, double angle)
    {
    QVector3D origine(*pOrigine);
    QVector3D direction(*p - *pOrigine);

    Ray3D rayon(origine, direction);
    double shortestDistance =  100000; // max int

    Face *bonneFace = 0;
    QVector3D *bonneIntersection = 0;

    foreach (Face *face, listFace) {
        Plane3D plan(*face->getList()->at(0), *face->getList()->at(1), *face->getList()->at(2));
        qreal t = plan.intersection(rayon);

        if(!qIsNaN(t))
        {
            QVector3D intersection = rayon.point(t);
            double distance = (intersection - origine).length();

            if(distance < shortestDistance)
            {
                shortestDistance = distance;

                if(bonneIntersection != 0)
                   delete bonneIntersection;

                bonneIntersection = new QVector3D(intersection);

                bonneFace = face;
            }
        }
    }

    Stain *tache = new Stain(bonneFace, bonneIntersection, pOrigine, new QColor(ball->getColor()));
    emit signal_addTache(tache);
    }

void Wall::draw()
    {
    glEnableClientState(GL_VERTEX_ARRAY);

    this->buffer->bind();
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    this->buffer->release();

    this->buffer2->bind();
    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT,NULL);
    this->buffer2->release();
    glDisableClientState(GL_VERTEX_ARRAY);
    }

bool Wall::isColliding(Collidable_A *other)
    {
    double arreteBoiteEnglobante = 0.2;
    double pxmin = other->getP()->x() - arreteBoiteEnglobante;
    double pzmin = other->getP()->z() - arreteBoiteEnglobante;
    double pymin = other->getP()->y() - arreteBoiteEnglobante;
    double pxmax = other->getP()->x() + arreteBoiteEnglobante;
    double pzmax = other->getP()->z() + arreteBoiteEnglobante;
    double pymax = other->getP()->y() + arreteBoiteEnglobante;

    double wxmin = this->p->x();
    double wzmin = this->p->z();
    double wxmax = this->p2->x();
    double wzmax = this->p4->z();
    double wymin = 0;
    double wymax = 20;

    if(pxmin < wxmax && pxmax > wxmin)
        {
        if(pzmax>wzmin && pzmin<wzmax)
            {
            if(pymin < wymax && pymax > wymin)
                {
                return true;
                }
            }
        }
    return false;
    }

Wall::~Wall()
    {
    try
    {
        qDeleteAll(listFace.begin(), listFace.end());
    }
    catch(...)
    {
        qDebug() << "destructeur de wall erreur";
    }
    }
