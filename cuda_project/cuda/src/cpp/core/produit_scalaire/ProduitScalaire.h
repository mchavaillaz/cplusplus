#ifndef PRODUITSCALAIRE_H_
#define PRODUITSCALAIRE_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class ProduitScalaire
    {
    public:
	ProduitScalaire(int _n);
	virtual ~ProduitScalaire(){}

	void launchProduitScalaire();
	float getResultat();

    private:
	int n;
	float resultat;
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
