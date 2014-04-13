#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include "Chronos.h"
#include "pi_entrelacement_promotion_tab_01.h"
#include "pi_entrelacement_critical_02.h"
#include "pi_entrelacement_atomic_03.h"
#include "pi_for_atomic_04.h"
#include "pi_for_critical_05.h"
#include "pi_for_promotion_tab_06.h"
#include "pi_for_automatic_07.h"
#include "produit_scalaire_entrelacement_promotion_tab_01.h"
#include "produit_scalaire_entrelacement_critical_02.h"
#include "produit_scalaire_entrelacement_atomic_03.h"
#include "produit_scalaire_for_atomic_04.h"
#include "produit_scalaire_for_critical_05.h"
#include "produit_scalaire_for_promotion_tab_06.h"
#include "produit_scalaire_for_automatic_07.h"
#include "monte_carlo_entrelacement_atomic.h"
#include "monte_carlo_for_atomic.h"
#include "monte_carlo_entrelacement_promotion_tab.h"
#include "histogramme_for_atomic.h"
#include "histogramme_entrelacement_promotion_tab.h"
#include "histogramme_entrelacement_atomic.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern void helloOMP1(void);
extern void helloOMP2(void);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainCore(void);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static bool useHello(void);
static bool usePi(void);
static bool useProduitScalaire(void);
static bool useMonteCarlo(void);
static void useHistogramme(void);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainCore(void)
    {
    bool isOk = true;
    Chronos chrono;
    chrono.start();

//    isOk &= useHello();
//    isOk &= usePi();
//    isOk &= useProduitScalaire();
//    isOk &= useMonteCarlo();
    useHistogramme();

    cout << "\n-------------------------------------------------" << endl;
    cout << "End Main : isOk = " << isOk << endl;

    return isOk ? EXIT_SUCCESS : EXIT_FAILURE;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

bool useHello(void)
    {
    helloOMP1();
    helloOMP2();

    return true;
    }

bool usePi(void)
    {
    int n = 1000;
    bool isOk = true;

    isOk &= pi_omp_entrelacement_promotion_tab_ok(n);
    isOk &= pi_omp_entrelacement_critical_ok(n);
    isOk &= pi_omp_entrelacement_atomic_ok(n);
    isOk &= pi_omp_for_atomic_ok(n);
    isOk &= pi_omp_for_automatic_ok(n);
    isOk &= pi_omp_for_critical_ok(n);
    isOk &= pi_omp_for_promotion_tab_ok(n);

    return isOk;
    }

bool useProduitScalaire(void)
    {
    int n = 16*1000;
    bool isOk = true;

    isOk &= produit_scalaire_omp_entrelacement_promotion_tab_ok(n);
//    isOk &= produit_scalaire_omp_entrelacement_critical_ok(n);
//    isOk &= produit_scalaire_omp_entrelacement_atomic_ok(n);
//    isOk &= produit_scalaire_omp_for_atomic_ok(n);
//    isOk &= produit_scalaire_omp_for_critical_ok(n);
//    isOk &= produit_scalaire_omp_for_promotion_tab_ok(n);
//    isOk &= produit_scalaire_omp_for_automatic_ok(n);

    return isOk;
    }

bool useMonteCarlo()
    {
    int n = 1000000;
    bool isOk = true;

//    isOk &= monte_carlo_omp_entrelacement_atomic_ok(n);
//    isOk &= monte_carlo_omp_for_atomic_ok(n);
    isOk &= monte_carlo_omp_entrelacement_promotion_tab_ok(n);

    return isOk;
    }

void useHistogramme()
    {
    int n = 1000;
//    launch_histogramme_omp_for_atomic(n);
    launch_histogramme_omp_entrelacement_promotion_tab(n);
//    launch_histogramme_omp_entrelacement_atomic(n);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

