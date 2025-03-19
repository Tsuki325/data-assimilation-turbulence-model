/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2020 OpenCFD Ltd.
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
#include <cmath>  
#include "kOmega.H"
#include "fvOptions.H"
#include "bound.H"
#include "tensorflow/c/c_api.h"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void kOmega<BasicTurbulenceModel>::correctNut()
{
    this->nut_ = k_/omega_;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmega<BasicTurbulenceModel>::kOmega
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
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

    Cmu_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    beta_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta",
            this->coeffDict_,
            0.072
        )
    ),
    gamma_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma",
            this->coeffDict_,
            0.52
        )
    ),
    alphaK_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega",
            this->coeffDict_,
            0.5
        )
    ),
    cor
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "cor",
            this->coeffDict_,
            1
        )
    ),
    begin
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "begin",
            this->coeffDict_,
            0
        )
    ),
    end
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "end",
            this->coeffDict_,
            1
        )
    ),
    height1
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "height1",
            this->coeffDict_,
            0
        )
    ),
    height2
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "height2",
            this->coeffDict_,
            0.1
        )
    ),
    ybegin
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "ybegin",
            this->coeffDict_,
            0.1
        )
    ),
    
    betannmodel
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betannmodel",
            this->coeffDict_,
            0
        )
    ),
    y_(wallDist::New(this->mesh_).y()),
    betann
    (
        IOobject
        (
            "betann", 
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

    gp
    (
        IOobject
        (
            "gp", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    
    Q
    (
        IOobject
        (
            "Q", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    Rew
    (
        IOobject
        (
            "Rew", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    lamda1_
    (
        IOobject
        (
            "lamda1", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    
    sw
    (
        IOobject
        (
            "sw", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),       
    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    )
       
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool kOmega<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        beta_.readIfPresent(this->coeffDict());
        gamma_.readIfPresent(this->coeffDict());
        alphaK_.readIfPresent(this->coeffDict());
        alphaOmega_.readIfPresent(this->coeffDict());
        cor.readIfPresent(this->coeffDict());
        begin.readIfPresent(this->coeffDict());
        end.readIfPresent(this->coeffDict());
        height1.readIfPresent(this->coeffDict());
        height2.readIfPresent(this->coeffDict());
        ybegin.readIfPresent(this->coeffDict());
        betannmodel.readIfPresent(this->coeffDict());
        //ifmodel.readIfPresent(this->coeffDict());
        return true;
    }

    return false;
}

void NoOpDeallocator(void* data, size_t a, void* b) {}
template<class BasicTurbulenceModel>
void kOmega<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    //const volScalarField& P = this->p_;
    //const volScalarField& p = mesh().p();

    const volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))().v()
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volScalarField::Internal GbyNu
    (
        tgradU().v() && dev(twoSymm(tgradU().v()))
    );
    const volScalarField::Internal G(this->GName(), nut()*GbyNu);
    tgradU.clear();

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();
    /*
    labelList renumberedCellLabels(this->mesh_.nCells(), -1); // 创建存储重新编号单元索引的数组
    forAll(this->mesh_.boundaryMesh(), patchI)
    {
    const labelList& patchCellLabels = this->mesh_.boundaryMesh()[patchI].faceCells();
    forAll(patchCellLabels, cellI)
    {
        label originalCellI = patchCellLabels[cellI];
        renumberedCellLabels[originalCellI] = cellI; // 存储重新编号后的单元索引
    }
    }
    */
    dimensioned<double> ifmodel = 0;
  if(betannmodel < ifmodel){
        const Foam::vectorField& cellCentres = this->mesh_.cellCentres();
        forAll(betann, cellI)
          {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
      if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )        
        {
            betann[cellI] = betann[cellI];
        }
        else
        {
            betann[cellI] = 1;
        }
       } 

    }
    // Turbulence specific dissipation rate equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhi, omega_)
      - fvm::laplacian(alpha*rho*DomegaEff(), omega_)
     ==
        gamma_*alpha()*rho()*GbyNu
      - fvm::SuSp(((2.0/3.0)*gamma_)*alpha()*rho()*divU, omega_)
      - betann*fvm::Sp(beta_*alpha()*rho()*omega_(), omega_)
      + fvOptions(alpha, rho, omega_)
    );

    omegaEqn.ref().relax();
    fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    fvOptions.correct(omega_);
    bound(omega_, this->omegaMin_);


    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(), k_)
     ==
        alpha()*rho()*G
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(Cmu_*alpha()*rho()*omega_(), k_)
      + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    fvOptions.correct(k_);
    bound(k_, this->kMin_);
    double currentTime = this->runTime_.time().value();
    double endTime = this->runTime_.endTime().value();
    if(betannmodel > ifmodel)
    {
    //////////////nn model
    /*   labelList renumberedCellLabels(this->mesh_.nCells(), -1); 
    forAll(this->mesh_.boundaryMesh(), patchI)
    {
    const labelList& patchCellLabels = this->mesh_.boundaryMesh()[patchI].faceCells();
    forAll(patchCellLabels, cellI)
    {
        label originalCellI = patchCellLabels[cellI];
        renumberedCellLabels[originalCellI] = cellI;
    }
    }
    */
    volScalarField::Internal disspationrate_(Cmu_*omega_()*k_);
    volScalarField::Internal production_(G);
    volScalarField::Internal pd(G/disspationrate_);
    volScalarField nu_(this->mu()/rho);
    volScalarField magw(mag(skew(fvc::grad(this->U_))));
    volScalarField mags(mag(symm(fvc::grad(this->U_))));
    Q = ( (sqr(magw)-sqr(mags)) / (sqr(magw)+sqr(mags)) );
    Rew = (magw*y_*y_/(nu_+this->nut_));
    //volScalarField svt(   (y_*mag(this->U_))/this->nut_ );
    sw = (mags/(magw+mags));
    volScalarField timeScale_(1/(0.09*omega_));
    volSymmTensorField S(timeScale_*symm((fvc::grad(this->U_))));
    volTensorField W(timeScale_*skew((fvc::grad(this->U_))));
    lamda1_ = (tr(W&W));
    volScalarField lamda2_(tr(S&S));
    volScalarField lamda3_(tr(W&W&S&S));
    volScalarField theta1(tr(S&S));
    volScalarField theta2(tr(S&S));
    volScalarField theta3(tr(S&S)); 
    volScalarField theta4(tr(S&S)); 
    scalar lamda1_minVal = 1e8;
    scalar lamda1_maxVal = -1e8;
    scalar sw_minVal = 1e8;
    scalar sw_maxVal = -1e8;
    scalar Q_minVal = 1e8;
    scalar Q_maxVal = -1e8;
    scalar Rew_minVal = 1e8;
    scalar Rew_maxVal = -1e8;
    const Foam::vectorField& cellCentres = this->mesh_.cellCentres();
    //弦长
    //double cor = 1;
    //double begin = 1;
    //double end = 1;
    //double height = 1;
    forAll(lamda1_, cellI)
    {
      //forAll(this->mesh_.boundaryMesh(), patchI)
       //{
          //const vector& coord = this->mesh_.C().boundaryField()[patchI][cellI];
          //const vector& coord = this->mesh_.C().boundaryField()[patchI][cellI];
          //scalar x = coord.x();
          //scalar y = y_[cellI];
          //scalar yy = coord.y();
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )
          {
             lamda1_minVal = min(lamda1_[cellI], lamda1_minVal);
             lamda1_maxVal = max(lamda1_[cellI], lamda1_maxVal);
             sw_minVal = min(sw[cellI], sw_minVal);
             sw_maxVal = max(sw[cellI], sw_maxVal);
             Q_minVal = min(Q[cellI],Q_minVal);
             Q_maxVal = max(Q[cellI],Q_maxVal);
             Rew_minVal = min(Rew[cellI], Rew_minVal);
             Rew_maxVal = max(Rew[cellI], Rew_maxVal);
          }
     //}
     }
     

        int nCells = this->mesh_.nCells();
        int nScalarInvariants = 4;
    // update theta
        for(int i=0;i<nCells;i++)
        {
           theta1[i] = lamda1_[i];
           theta2[i] = sw[i];
           theta3[i] = Rew[i];
           theta4[i] = Q[i];
        }
    //NORMALIZE
        scalar theta1_minVal = lamda1_minVal;
        scalar theta1_maxVal = lamda1_maxVal;
        scalar theta2_minVal = sw_minVal;
        scalar theta2_maxVal = sw_maxVal;
        scalar theta3_minVal = Rew_minVal;
        scalar theta3_maxVal = Rew_maxVal; 
        scalar theta4_minVal = Q_minVal;
        scalar theta4_maxVal = Q_maxVal; 
   
    //NORMALIZE      
        forAll(theta1, cellI)
        {
        theta1[cellI] = (theta1[cellI]-theta1_minVal)/(theta1_maxVal-theta1_minVal);
        theta2[cellI] = (theta2[cellI]-theta2_minVal)/(theta2_maxVal-theta2_minVal);
        theta3[cellI] = (theta3[cellI]-theta3_minVal)/(theta3_maxVal-theta3_minVal);
        theta4[cellI] = (theta4[cellI]-theta4_minVal)/(theta4_maxVal-theta4_minVal);
        }
    
        #include "betann4in1out_.H"
        //const Foam::vectorField& cellCentres = this->mesh_.cellCentres();
        forAll(betann, cellI)
          {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
      if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )        
        {
            betann[cellI] = betann[cellI];
        }
        else
        {
            betann[cellI] = 1;
        }
       }
    }
        

    // end nn model
    
    
    //write feature
    
    if (currentTime==endTime)
    {
    volScalarField::Internal disspationrate_(Cmu_*omega_()*k_);
    volScalarField::Internal production_(G);
    volScalarField::Internal pd(G/disspationrate_);
    volScalarField::Internal gp1(fvc::grad(p) & U);
    volScalarField::Internal gp2(mag(fvc::grad(p))*mag(U));
    volScalarField nu_(this->mu()/rho);
    volScalarField magw(mag(skew(fvc::grad(this->U_))));
    volScalarField mags(mag(symm(fvc::grad(this->U_))));

    Q =  (sqr(magw)-sqr(mags)) / (sqr(magw)+sqr(mags)) ;
    Rew = magw*y_*y_ / (nu_+this->nut_);
    //volScalarField svt(   (y_*mag(this->U_))/this->nut_ );
    sw = mags/(magw+mags);
    volScalarField timeScale_(1/(0.09*omega_));
    gp = gp1/gp2;
    volSymmTensorField S(timeScale_*symm((fvc::grad(this->U_))));
    volTensorField W(timeScale_*skew((fvc::grad(this->U_))));
    volScalarField magwhat(mag(W));
    volScalarField magshat(mag(S));
    lamda1_ = tr(W&W);
    volScalarField lamda2_(tr(S&S));
    volScalarField lamda3_(tr(W&W&S&S));
    volScalarField theta1(tr(S&S));
    volScalarField theta2(tr(S&S));
    volScalarField theta3(tr(S&S)); 
    volScalarField theta4(tr(S&S)); 
    scalar lamda1_minVal = 1e8;
    const Foam::vectorField& cellCentres = this->mesh_.cellCentres();
    //弦长
    //double cor = 1;
    //double begin = 1;
    //double end = 1;
    //double height = 1;
    forAll(lamda1_, cellI)
    {
      //forAll(this->mesh_.boundaryMesh(), patchI)
       //{
          //const vector& coord = this->mesh_.C().boundaryField()[patchI][cellI];
          //const vector& coord = this->mesh_.C().boundaryField()[patchI][cellI];
          //scalar x = coord.x();
          //scalar y = y_[cellI];
          //scalar yy = coord.y();
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )
          {
             lamda1_minVal = min(lamda1_[cellI], lamda1_minVal);
          }
     //}
     }
     
    forAll(lamda1_, cellI)
    {
      //forAll(this->mesh_.boundaryMesh(), patchI){
          //const vector& coord = this->mesh_.C().boundaryField()[patchI][cellI];
          //const vector& coord = this->mesh_.C().boundaryField()[patchI][cellI];[cellI];
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )      
             {
               lamda1_[cellI] = lamda1_[cellI];
             }
          else
             {
                lamda1_[cellI] = lamda1_minVal;
             }
        //}
    }     

    scalar lamda2_minVal = 1e8;
    forAll(lamda2_, cellI)
    {
      //forAll(this->mesh_.boundaryMesh(), patchI)
       //{
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )  
          {
             lamda2_minVal = min(lamda2_[cellI], lamda2_minVal);
          }
        
     //}
     }
  
    forAll(lamda2_, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )  
          {
               lamda2_[cellI] = lamda2_[cellI];
             }
          else{
                lamda2_[cellI] = lamda2_minVal;
              }
    }
    
    //
    
    scalar lamda3_minVal = 1e8;
    forAll(lamda3_, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )  
          {
             lamda3_minVal = min(lamda3_[cellI], lamda3_minVal);
          } 
     }

    forAll(lamda3_, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin ) 
             {
               lamda3_[cellI] = lamda3_[cellI];
             }
          else{
                lamda3_[cellI] = lamda3_minVal;
             }
     }



    scalar Rew_minVal = 1e8;
    forAll(Rew, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin ) 
          { 
             Rew_minVal = min(Rew[cellI], Rew_minVal);
          } 
    }
    forAll(Rew, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )  
             {
               Rew[cellI] = Rew[cellI];
             }
          else
             {
                Rew[cellI]= Rew_minVal;
             }
     }
 
 
    scalar pd_minVal = 1e8;
    forAll(pd, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )  
          {
             pd_minVal = min(pd[cellI], pd_minVal);
          } 
     }
         
    forAll(pd, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin ) 
             { 
               pd[cellI] = pd[cellI];
             }
          else
             {
                pd[cellI] = pd_minVal;
             }
    }

    scalar Q_minVal = 1e8;
    forAll(Q, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin ) 
          { 
             Q_minVal = min(Q[cellI], Q_minVal);
          } 
    }
    forAll(Q, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )  
             {
               Q[cellI] = Q[cellI];
             }
          else
             {
                Q[cellI]= Q_minVal;
             }
     }
     
         
    scalar sw_minVal = 1e8;
    forAll(sw, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )  
          {
             sw_minVal = min(sw[cellI], sw_minVal);
          } 
         
     }
         
    forAll(sw, cellI)
    {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
          if ( x > begin*cor && x < end*cor && d < height2*cor && d > height1*cor  && y > ybegin )        
             {
               sw[cellI] = sw[cellI];
             }
          else{
                sw[cellI] = sw_minVal;
             }
    }
     
    forAll(gp, cellI)
    {
         gp[cellI]=std::acos(gp[cellI])*2/3.14-1;
    }

    
    sw.write();
    //pd.write();
    Rew.write();
    //magw.write();
    //magwhat.write();
    //magshat.write();
    lamda1_.write();
    //lamda2_.write();
    //lamda3_.write();
    Q.write();
    gp.write();
    //svt.write();
    }
    //end write feature
    
    correctNut();
}
  

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
