/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2016-2020 OpenCFD Ltd.
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

#include "kOmegaSSTBase.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
void NoOpDeallocator(void* data, size_t a, void* b) {}
namespace Foam
{

// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBase<BasicEddyViscosityModel>::F1
(
    const volScalarField& CDkOmega
) const
{
    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar("1.0e-10", dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
            ),
            (4*alphaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBase<BasicEddyViscosityModel>::F2() const
{
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBase<BasicEddyViscosityModel>::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBase<BasicEddyViscosityModel>::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBase<BasicEddyViscosityModel>::correctNut
(
    const volScalarField& S2
)
{
    // Correct the turbulence viscosity
    this->nut_ = a1_*k_/max(a1_*omega_, b1_*F23()*sqrt(S2));
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBase<BasicEddyViscosityModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBase<BasicEddyViscosityModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*this->k_()*this->omega_());
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal>
kOmegaSSTBase<BasicEddyViscosityModel>::epsilonByk
(
    const volScalarField& F1,
    const volTensorField& gradU
) const
{
    return betaStar_*omega_();
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBase<BasicEddyViscosityModel>::GbyNu
(
    const volScalarField::Internal& GbyNu0,
    const volScalarField::Internal& F2,
    const volScalarField::Internal& S2
) const
{
    return min
    (
        GbyNu0,
        (c1_/a1_)*betaStar_*omega_()
       *max(a1_*omega_(), b1_*F2*sqrt(S2))
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBase<BasicEddyViscosityModel>::kSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            k_,
            dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
        )
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBase<BasicEddyViscosityModel>::omegaSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
        )
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBase<BasicEddyViscosityModel>::Qsas
(
    const volScalarField::Internal& S2,
    const volScalarField::Internal& gamma,
    const volScalarField::Internal& beta
) const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
        )
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
kOmegaSSTBase<BasicEddyViscosityModel>::kOmegaSSTBase
(
    const word& type,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    BasicEddyViscosityModel
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
    

    alphaK1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK1",
            this->coeffDict_,
            0.85
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK2",
            this->coeffDict_,
            1.0
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega1",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega2",
            this->coeffDict_,
            0.856
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma1",
            this->coeffDict_,
            5.0/9.0
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma2",
            this->coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta1",
            this->coeffDict_,
            0.075
        )
    ),
    beta2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta2",
            this->coeffDict_,
            0.0828
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    a1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "a1",
            this->coeffDict_,
            0.31
        )
    ),
    b1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "b1",
            this->coeffDict_,
            1.0
        )
    ),
    c1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "c1",
            this->coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::getOrAddToDict
        (
            "F3",
            this->coeffDict_,
            false
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
    
    Main
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "Main",
            this->coeffDict_,
            2.84
        )
    ),
    
    betascalar
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betascalar",
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
    nk
    (
        IOobject
        (
            "nk", 
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

    Res
    (
        IOobject
        (
            "Res", 
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
    
    I1
    (
        IOobject
        (
            "I1", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ), 
    
    I2
    (
        IOobject
        (
            "I2", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ), 
    
    q2
    (
        IOobject
        (
            "q2", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ), 
    
    
    q3
    (
        IOobject
        (
            "q3", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ), 
    lamda2_
    (
        IOobject
        (
            "lamda2", 
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
    TuM
    (
        IOobject
        (
            "TuM", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),  
    Mg
    (
        IOobject
        (
            "Mg", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ), 
    
    xx
    (
        IOobject
        (
            "xx", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ), 

    Fs
    (
        IOobject
        (
            "Fs", 
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),   
    
    PD
    (
        IOobject
        (
            "PD", 
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
    ),

    decayControl_
    (
        Switch::getOrAddToDict
        (
            "decayControl",
            this->coeffDict_,
            false
        )
    ),
    kInf_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "kInf",
            this->coeffDict_,
            k_.dimensions(),
            0
        )
    ),
    omegaInf_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "omegaInf",
            this->coeffDict_,
            omega_.dimensions(),
            0
        )
    )
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);

    setDecayControl(this->coeffDict_);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
void kOmegaSSTBase<BasicEddyViscosityModel>::setDecayControl
(
    const dictionary& dict
)
{
    decayControl_.readIfPresent("decayControl", dict);

    if (decayControl_)
    {
        kInf_.read(dict);
        omegaInf_.read(dict);

        Info<< "    Employing decay control with kInf:" << kInf_
            << " and omegaInf:" << omegaInf_ << endl;
    }
    else
    {
        kInf_.value() = 0;
        omegaInf_.value() = 0;
    }
}


template<class BasicEddyViscosityModel>
bool kOmegaSSTBase<BasicEddyViscosityModel>::read()
{
    if (BasicEddyViscosityModel::read())
    {
        alphaK1_.readIfPresent(this->coeffDict());
        alphaK2_.readIfPresent(this->coeffDict());
        alphaOmega1_.readIfPresent(this->coeffDict());
        alphaOmega2_.readIfPresent(this->coeffDict());
        gamma1_.readIfPresent(this->coeffDict());
        gamma2_.readIfPresent(this->coeffDict());
        beta1_.readIfPresent(this->coeffDict());
        beta2_.readIfPresent(this->coeffDict());
        betaStar_.readIfPresent(this->coeffDict());
        a1_.readIfPresent(this->coeffDict());
        b1_.readIfPresent(this->coeffDict());
        c1_.readIfPresent(this->coeffDict());
        F3_.readIfPresent("F3", this->coeffDict());

        setDecayControl(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBase<BasicEddyViscosityModel>::correct()
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
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    BasicEddyViscosityModel::correct();

    volScalarField::Internal divU(fvc::div(fvc::absolute(this->phi(), U)));

    tmp<volTensorField> tgradU = fvc::grad(U);
    volScalarField S2(2*magSqr(symm(tgradU())));
    volScalarField::Internal GbyNu0
    (
        this->type() + ":GbyNu",
        (tgradU() && dev(twoSymm(tgradU())))
    );
    volScalarField::Internal G(this->GName(), nut*GbyNu0);

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();

    volScalarField CDkOmega
    (
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    volScalarField F1(this->F1(CDkOmega));
    volScalarField F23(this->F23());   
    {
        volScalarField::Internal gamma(this->gamma(F1));
        volScalarField::Internal beta(this->beta(F1));

        GbyNu0 = GbyNu(GbyNu0, F23(), S2());

/*

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
      if ( d < height2*cor )        
        {
            betann[cellI] = betann[cellI];
        }
        else
        {
            betann[cellI] = 1;
        }
       } 

    }

   
        // Turbulent frequency equation
        tmp<fvScalarMatrix> omegaEqn
        (
            fvm::ddt(alpha, rho, omega_)
          + fvm::div(alphaRhoPhi, omega_)
          - fvm::laplacian(alpha*rho*DomegaEff(F1), omega_)
         ==
            alpha()*rho()*gamma*GbyNu0
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*gamma*divU, omega_)
          - fvm::Sp(betann*alpha()*rho()*beta*omega_(), omega_)
          - fvm::SuSp
            (
                alpha()*rho()*(F1() - scalar(1))*CDkOmega()/omega_(),
                omega_
            )
          + alpha()*rho()*beta*sqr(omegaInf_)
          + Qsas(S2(), gamma, beta)
          + omegaSource()
          + fvOptions(alpha, rho, omega_)
        );

        omegaEqn.ref().relax();
        fvOptions.constrain(omegaEqn.ref());
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
        solve(omegaEqn);
        fvOptions.correct(omega_);
        bound(omega_, this->omegaMin_);
    }

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(F1), k_)
     ==
        alpha()*rho()*Pk(G)
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(alpha()*rho()*epsilonByk(F1, tgradU()), k_)
      + alpha()*rho()*betaStar_*omegaInf_*kInf_
      + kSource()
      + fvOptions(alpha, rho, k_)
    );

    tgradU.clear();

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    fvOptions.correct(k_);
    bound(k_, this->kMin_);
    double currentTime = this->runTime_.time().value();
    double endTime = this->runTime_.endTime().value();

    dimensioned<double> ifmodel = 0;
    double ifmodeltime = 500 ;
    if(betannmodel > ifmodel and currentTime > ifmodeltime)
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
 
    volScalarField nu_(this->mu()/this->rho_);
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
          if ( d < height2*cor )
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
    
        #include "betann4in1out.H"
        //const Foam::vectorField& cellCentres = this->mesh_.cellCentres();
        forAll(betann, cellI)
          {
          const Foam::vector& cellCentre = cellCentres[cellI];
          dimensioned<scalar> x = cellCentre.x();
          dimensioned<scalar> y = cellCentre.y();
          dimensioned<scalar> d = y_[cellI];
          //if (x > 0  && x < 0.5 && y > 0 && y < 0.008 && yy > 0)
      if (d < height2*cor)        
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
    
    //if (currentTime==endTime)
    //{
   
    
    correctNut(S2);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
