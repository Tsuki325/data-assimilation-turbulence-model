/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2017-2019 OpenCFD Ltd.
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

Class
    Foam::kOmegaSSTBase

Description
    Base class implementation of the k-omega-SST turbulence model for
    incompressible and compressible flows.

    Turbulence model described in:
    \verbatim
        Menter, F. R. & Esch, T. (2001).
        Elements of Industrial Heat Transfer Prediction.
        16th Brazilian Congress of Mechanical Engineering (COBEM).
    \endverbatim

    with updated coefficients from
    \verbatim
        Menter, F. R., Kuntz, M., and Langtry, R. (2003).
        Ten Years of Industrial Experience with the SST Turbulence Model.
        Turbulence, Heat and Mass Transfer 4, ed: K. Hanjalic, Y. Nagano,
        & M. Tummers, Begell House, Inc., 625 - 632.
    \endverbatim

    but with the consistent production terms from the 2001 paper as form in the
    2003 paper is a typo, see
    \verbatim
        http://turbmodels.larc.nasa.gov/sst.html
    \endverbatim

    and the addition of the optional F3 term for rough walls from
    \verbatim
        Hellsten, A. (1998).
        Some Improvements in Menter's k-omega-SST turbulence model
        29th AIAA Fluid Dynamics Conference, AIAA-98-2554.
    \endverbatim

    and the optional decay control from:
    \verbatim
        Spalart, P. R. and Rumsey, C. L. (2007).
        Effective Inflow Conditions for Turbulence Models in Aerodynamic
        Calculations
        AIAA Journal, 45(10), 2544 - 2553.
    \endverbatim

    Note that this implementation is written in terms of alpha diffusion
    coefficients rather than the more traditional sigma (alpha = 1/sigma) so
    that the blending can be applied to all coefficients in a consistent
    manner.  The paper suggests that sigma is blended but this would not be
    consistent with the blending of the k-epsilon and k-omega models.

    Also note that the error in the last term of equation (2) relating to
    sigma has been corrected.

    Wall-functions are applied in this implementation by using equations (14)
    to specify the near-wall omega as appropriate.

    The blending functions (15) and (16) are not currently used because of the
    uncertainty in their origin, range of applicability and that if y+ becomes
    sufficiently small blending u_tau in this manner clearly becomes nonsense.

    The default model coefficients are
    \verbatim
        kOmegaSSTBaseCoeffs
        {
            alphaK1         0.85;
            alphaK2         1.0;
            alphaOmega1     0.5;
            alphaOmega2     0.856;
            beta1           0.075;
            beta2           0.0828;
            betaStar        0.09;
            gamma1          5/9;
            gamma2          0.44;
            a1              0.31;
            b1              1.0;
            c1              10.0;
            F3              no;

            // Optional decay control
            decayControl    yes;
            kInf            \<far-field k value\>;
            omegaInf        \<far-field omega value\>;
        }
    \endverbatim

SourceFiles
    kOmegaSSTBase.C

\*---------------------------------------------------------------------------*/

#ifndef kOmegaSSTBase_H
#define kOmegaSSTBase_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class kOmegaSSTBase Declaration
\*---------------------------------------------------------------------------*/

template<class BasicEddyViscosityModel>
class kOmegaSSTBase
:
    public BasicEddyViscosityModel
{
    // Private Member Functions

        //- No copy construct
        kOmegaSSTBase(const kOmegaSSTBase&) = delete;

        //- No copy assignment
        void operator=(const kOmegaSSTBase&) = delete;


protected:

    // Protected data

        // Model coefficients
            dimensionedScalar alphaK1_;
            dimensionedScalar alphaK2_;

            dimensionedScalar alphaOmega1_;
            dimensionedScalar alphaOmega2_;

            dimensionedScalar gamma1_;
            dimensionedScalar gamma2_;

            dimensionedScalar beta1_;
            dimensionedScalar beta2_;

            dimensionedScalar betaStar_;

            dimensionedScalar a1_;
            dimensionedScalar b1_;
            dimensionedScalar c1_;

            //- Flag to include the F3 term
            Switch F3_;
            //area
            dimensionedScalar cor;
            dimensionedScalar begin;
            dimensionedScalar end;
            dimensionedScalar Main;
            dimensionedScalar sca;
            dimensionedScalar betascalar;
            dimensionedScalar height1;
            dimensionedScalar height2;
            dimensionedScalar ybegin;
            dimensionedScalar betannmodel;

        // Fields

            //- Wall distance
            //  Note: different to wall distance in parent RASModel
            //  which is for near-wall cells only
            const volScalarField& y_;
            //volScalarField::Internal pd;     
            volScalarField betann; 
            volScalarField p;
            volScalarField V;
            volScalarField nk;
            volScalarField Rew;
            //volScalarField::Internal gp;
            //volScalarField::Internal xv;
            //volScalarField Q;
            volScalarField Res;
            volScalarField lamda1_;
            volScalarField I1;
            volScalarField I2;
            volScalarField q2;
            volScalarField q3;
            volScalarField lamda2_;
            volScalarField sw;
            volScalarField TuM;
            //volScalarField rd;
            volScalarField Mg;  
            volScalarField xx;
            volScalarField Fs;
            volScalarField PD;      
            volScalarField k_;
            volScalarField omega_;
            //volScalarField T_;
            
            //volScalarField::Internal production_;
            //volScalarField::Internal production;
            


        // Decay control

            //- Flag to include the decay control
            Switch decayControl_;
            dimensionedScalar kInf_;
            dimensionedScalar omegaInf_;
             //Mean strain rate



    // Protected Member Functions

        void setDecayControl(const dictionary& dict);

        virtual tmp<volScalarField> F1(const volScalarField& CDkOmega) const;
        virtual tmp<volScalarField> F2() const;
        virtual tmp<volScalarField> F3() const;
        virtual tmp<volScalarField> F23() const;

        tmp<volScalarField> blend
        (
            const volScalarField& F1,
            const dimensionedScalar& psi1,
            const dimensionedScalar& psi2
        ) const
        {
            return F1*(psi1 - psi2) + psi2;
        }

        tmp<volScalarField::Internal> blend
        (
            const volScalarField::Internal& F1,
            const dimensionedScalar& psi1,
            const dimensionedScalar& psi2
        ) const
        {
            return F1*(psi1 - psi2) + psi2;
        }

        tmp<volScalarField> alphaK(const volScalarField& F1) const
        {
            return blend(F1, alphaK1_, alphaK2_);
        }

        tmp<volScalarField> alphaOmega(const volScalarField& F1) const
        {
            return blend(F1, alphaOmega1_, alphaOmega2_);
        }

        tmp<volScalarField::Internal> beta
        (
            const volScalarField::Internal& F1
        ) const
        {
            return tmp<volScalarField::Internal>::New
            (
                this->type() + ":beta",
                blend(F1, beta1_, beta2_)
            );
        }

        tmp<volScalarField::Internal> gamma
        (
            const volScalarField::Internal& F1
        ) const
        {
            return tmp<volScalarField::Internal>::New
            (
                this->type() + ":gamma",
                blend(F1, gamma1_, gamma2_)
            );
        }

        virtual void correctNut(const volScalarField& S2);

        virtual void correctNut();

        //- Return k production rate
        virtual tmp<volScalarField::Internal> Pk
        (
            const volScalarField::Internal& G
        ) const;

        //- Return epsilon/k which for standard RAS is betaStar*omega
        virtual tmp<volScalarField::Internal> epsilonByk
        (
            const volScalarField& F1,
            const volTensorField& gradU
        ) const;

        //- Return G/nu
        virtual tmp<volScalarField::Internal> GbyNu
        (
            const volScalarField::Internal& GbyNu0,
            const volScalarField::Internal& F2,
            const volScalarField::Internal& S2
        ) const;

        virtual tmp<fvScalarMatrix> kSource() const;

        virtual tmp<fvScalarMatrix> omegaSource() const;

        virtual tmp<fvScalarMatrix> Qsas
        (
            const volScalarField::Internal& S2,
            const volScalarField::Internal& gamma,
            const volScalarField::Internal& beta
        ) const;


public:

    typedef typename BasicEddyViscosityModel::alphaField alphaField;
    typedef typename BasicEddyViscosityModel::rhoField rhoField;
    typedef typename BasicEddyViscosityModel::transportModel transportModel;


    // Constructors

        //- Construct from components
        kOmegaSSTBase
        (
            const word& type,
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName
        );


    //- Destructor
    virtual ~kOmegaSSTBase() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read();

        //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff(const volScalarField& F1) const
        {
            return tmp<volScalarField>
            (
                new volScalarField("DkEff", alphaK(F1)*this->nut_ + this->nu())
            );
        }

        //- Return the effective diffusivity for omega
        tmp<volScalarField> DomegaEff(const volScalarField& F1) const
        {
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    "DomegaEff",
                    alphaOmega(F1)*this->nut_ + this->nu()
                )
            );
        }

        //- Return the turbulence kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        //- Return the turbulence kinetic energy dissipation rate
        virtual tmp<volScalarField> epsilon() const
        {
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    IOobject
                    (
                        "epsilon",
                        this->mesh_.time().timeName(),
                        this->mesh_
                    ),
                    betaStar_*k_*omega_,
                    omega_.boundaryField().types()
                )
            );
        }

        //- Return the turbulence kinetic energy dissipation rate
        virtual tmp<volScalarField> omega() const
        {
            return omega_;
        }

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "kOmegaSSTBase.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
