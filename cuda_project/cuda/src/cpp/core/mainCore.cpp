#include <iostream>
#include <stdlib.h>
#include "limits.h"
#include "Device.h"
#include "cudaTools.h"
#include "ProduitScalaire.h"
#include "FonctionsProduitScalaire.h"
#include "MonteCarloHost.h"
#include "Chronos.h"
#include "AleaTools.h"
#include "FonctionsMonteCarlo.h"
#include "HistogrammeHost.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern void helloCuda(void);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainCore(int deviceId);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static bool useHello(void);
static void useProduitScalaire(void);
static void useMonteCarlo(void);
static void useHistogramme(void);

static float produitScalaireSequentielleCuda(int n);
static double monteCarloSequentielleCuda(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainCore(int deviceId)
    {
    Device::print(deviceId, "Execute on device : ");

    bool isOk = true;

//    useHistogramme(); --> je suis tout prêt de la solution :)
    useMonteCarlo();
    cout << endl;
    useProduitScalaire();

    cout << "\nEnd : mainCore" << endl;

    return isOk ? EXIT_SUCCESS : EXIT_FAILURE;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

bool useHello(void)
    {
    helloCuda();
    return true;
    }

void useProduitScalaire(void)
    {
    int n = 8192;

    Chronos chronoCuda;

    chronoCuda.start();

    ProduitScalaire *produitScalaire = new ProduitScalaire(n);
    produitScalaire->launchProduitScalaire();

    double secondesCuda = chronoCuda.stop();

    Chronos chronoCPU;

    chronoCPU.start();

    float resultatSequentiel = produitScalaireSequentielleCuda(n);

    double secondesCPU = chronoCPU.stop();

    cout << "Resultat produit scalaire cuda= " << produitScalaire->getResultat() << " temps d'execution= " << secondesCuda << endl;
    cout << "Resultat produit scalaire sequentiel= " << resultatSequentiel << " temps d'execution= " << secondesCPU << endl;

    delete produitScalaire;
    }

static void useHistogramme(void)
    {
    Chronos chronoHistogramme;

    chronoHistogramme.start();

    Histogramme *histogramme = new Histogramme();
    histogramme->run();

    double secondesHistogramme = chronoHistogramme.stop();

    histogramme->printTabFrequence();
    cout << "Temps d'execution= " << secondesHistogramme << endl;

    delete histogramme;
    }

float produitScalaireSequentielleCuda(int n)
    {
    float total = 0;

    for (int i = 0; i < n; i++)
	{
	total += vHost(i) * wHost(i);
	}

    return total;
    }

static double monteCarloSequentielleCuda(int n)
    {
    long sommeInTot = 0;
    long sommeOutTot = 0;
    float a = -1.0;
    float b = 1.0;
    float m = 8.5;
    AleaTools alea;

    for (long i = 0; i < n; i++)
	{
	double randX = alea.uniformeAB(a, b);
	double randY = alea.uniformeAB(0.0, m);

	if (randY <= fHost(randX))
	    {
	    sommeInTot++;
	    }
	else
	    {
	    sommeOutTot++;
	    }
	}

    double numerateur = (b - a) * m * sommeInTot;
    double denominateur = n;
    return (numerateur / denominateur) / 2;
    }

void useMonteCarlo(void)
    {
    int n = 8192;
    float a = -1.0;
    float b = 1.0;
    float m = 8.5;

    Chronos chronoCudaMonteCarlo;

    chronoCudaMonteCarlo.start();

    MonteCarlo *monteCarlo = new MonteCarlo(a, b, m, n);
    monteCarlo->run();

    double secondesCudaMonteCarlo = chronoCudaMonteCarlo.stop();

    Chronos chronoMonteCarloSequentiel;

    float resultatSequentiel = monteCarloSequentielleCuda(n * 10000);

    double secondesMonteCarloSequentiel = chronoMonteCarloSequentiel.stop();

    cout << "Resultat monte carlo cuda= " << monteCarlo->getResultat() << " temps d'excecution = " << secondesCudaMonteCarlo << endl;
    cout << "Resultat monte carlo sequentielle= " << resultatSequentiel << " temps d'excecution = " << secondesMonteCarloSequentiel << endl;

    delete monteCarlo;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

