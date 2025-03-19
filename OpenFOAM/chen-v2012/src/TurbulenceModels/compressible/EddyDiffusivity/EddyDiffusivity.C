/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "EddyDiffusivity.H"
#include "fvOptions.H"
#include "fvc.H"
#include "tensorflow/c/c_api.h"
// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //
static void NoOpDeallocator1(void* data, size_t a, void* b) {}
template<class BasicTurbulenceModel>
void Foam::EddyDiffusivity<BasicTurbulenceModel>::correctNut()
{
    int nCells = this->mesh_.nCells();
    //Read Prt if provided
    Prtnnmodel = dimensionedScalar("Prtnnmodel", dimless,0, this->coeffDict());
    Prtempmodel = dimensionedScalar("Prtempmodel", dimless,0, this->coeffDict());
    Main_ = dimensionedScalar("Main", dimless,1, this->coeffDict());
    
    volScalarField magw(mag(-skew(fvc::grad(this->U_))));
    volScalarField mags(mag(symm(fvc::grad(this->U_))));
    volScalarField nk(this->k()/(this->k()+ 0.5*mag(this->U_)*mag(this->U_)));
    volScalarField xx(this->nut()/(this->mu()/this->rho_));
    volScalarField Fs = 0.5-0.5*Foam::tanh(50*tr(symm(fvc::grad(this->U_)))*(max( 0.25*sqrt(this->k())/0.09/this->omega(), pow(V,1.0/3.0)))/(sqrt(1.4*p/this->rho_))+5);
    volScalarField PD = Foam::tanh (max(this->nut()*mags*mags/ (0.09*this->omega()*this->k())-1.2,0.0));
    volScalarField Mg = mag(dev(symm(fvc::grad(this->U_))))*1.414*sqrt(this->k())/(this->omega()*sqrt(1.4*p/this->rho_));
    volScalarField TuM = sqrt(2.0*this->k()) / sqrt(1.4*p/this->rho_);
    volScalarField theta1(mags);
    volScalarField theta2(mags);
    volScalarField theta3(mags); 
    volScalarField theta4(mags); 
    volScalarField theta5(mags); 
    volScalarField theta6(mags); 
    double currentTime = this->runTime_.time().value();
    double ifmodeltime = 3000 ;
          //input of Prt_nn
    dimensioned<double> ifPrtmodel = 0;
    if(Prtnnmodel > ifPrtmodel and currentTime > ifmodeltime)
    {
    int nScalarInvariants=6;
          for(int i=0;i<nCells;i++)
          {
          theta1[i] = Fs[i];
          theta2[i] = nk[i];
          theta3[i] =  Main_.value();
          theta4[i] = PD[i];
          theta5[i] = Mg[i];
          theta6[i] = TuM[i]; 
          }
    #include "Prtnn6in1out.H"
    }
    
    if(Prtempmodel > ifPrtmodel)
    //Prt_ = 1;
    {
    //Prt_ = 1 - 0.8*Foam::tanh( 10.0 * max(this->nut()*mags*mags / (0.09*this->omega()*this->k())-1.2,0.0 ));
    Prt_ = 1 - 0.7*nk;
    }
    
    //Prt_ = 1;
    alphat_ = this->rho_*this->nut()/Prt_;
    alphat_.correctBoundaryConditions();

}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Foam::EddyDiffusivity<BasicTurbulenceModel>::EddyDiffusivity
(
    const word& type,
    const alphaField& alpha,
    const volScalarField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    BasicTurbulenceModel
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    // Cannot read Prt yet
    //Prt_("Prt", dimless, 1.0),
    Prt_
    (
        IOobject
        (
            "Prt", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),  
    
    V
    (
        IOobject
        (
            "V", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ), 
    p
    (
        IOobject
        (
            "p", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),   
                  
    alphat_
    (
        IOobject
        (
            IOobject::groupName("alphat", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    )  
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
template<class BasicTurbulenceModel>
bool Foam::EddyDiffusivity<BasicTurbulenceModel>::read()
{

}


template<class BasicTurbulenceModel>
void Foam::EddyDiffusivity<BasicTurbulenceModel>::correctEnergyTransport()
{
    EddyDiffusivity<BasicTurbulenceModel>::correctNut();
}


// ************************************************************************* //
