//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file DetectorConstruction.cc
/// \brief Implementation of the B1::DetectorConstruction class

#include "DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Orb.hh"
#include "G4Sphere.hh"
#include "G4Trd.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"
#include "CLHEP/Units/SystemOfUnits.h"

namespace B1
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // Get nist material manager
  G4NistManager* nist = G4NistManager::Instance();

  // Envelope parameters
  //
  G4double env_sizeXY = 20*cm, env_sizeZ = 30*cm;

  // Option to switch on/off checking of volumes overlaps
  //
  G4bool checkOverlaps = true;

  //
  // World
  //
  G4double world_sizeXY = 1.2*env_sizeXY;
  G4double world_sizeZ  = 1.2*env_sizeZ;
  G4Material* world_mat = nist->FindOrBuildMaterial("G4_AIR");

  G4Box* solidWorld =
    new G4Box("World",                       //its name
       0.5*world_sizeXY, 0.5*world_sizeXY, 0.5*world_sizeZ);     //its size

  G4LogicalVolume* logicWorld =
    new G4LogicalVolume(solidWorld,          //its solid
                        world_mat,           //its material
                        "World");            //its name

  G4VPhysicalVolume* physWorld =
    new G4PVPlacement(0,                     //no rotation
                      G4ThreeVector(),       //at (0,0,0)
                      logicWorld,            //its logical volume
                      "World",               //its name
                      0,                     //its mother  volume
                      false,                 //no boolean operation
                      0,                     //copy number
                      checkOverlaps);        //overlaps checking

  //
  // Blindage
  //
  G4Material* envelope_mat = nist->FindOrBuildMaterial("G4_Al");
  // Position du cylindre englobant (centre entre les deux zones)
  G4ThreeVector posEnvelope = G4ThreeVector(0, 0,0);

  // Dimensions du cylindre englobant
  G4double pRMinEnvelope = 0 * cm;
  G4double pRMaxEnvelope = 2.5 * cm;
  G4double pDzEnvelope = 2.5 * cm + 0.05 * cm + 0.5 * cm; // hauteur totale des deux zones + marges

  G4Tubs* solidEnvelope = 
      new G4Tubs("Envelope",
          pRMinEnvelope, pRMaxEnvelope, pDzEnvelope, 0, 2 * CLHEP::pi);

  G4LogicalVolume* logicEnvelope = 
      new G4LogicalVolume(solidEnvelope, envelope_mat, "Envelope");

  new G4PVPlacement(0, posEnvelope, logicEnvelope, "Envelope", logicWorld, false, 0, checkOverlaps);

  //
  // Zone active
  //
  G4Material* shape1_mat = nist->FindOrBuildMaterial("G4_Ge");
  G4ThreeVector pos1 = G4ThreeVector(0, 0, 0.1);

  G4double pRMin=0*cm;
  G4double pRMax=2*cm;
  G4double pDz=2.5*cm;
  G4double pSPhi=0;
  G4double pDPhi=2 * CLHEP::pi;
  G4Tubs* solidShape1 =
    new G4Tubs("Zone_active",
        pRMin, pRMax, pDz, pSPhi, pDPhi);

  G4LogicalVolume* logicShape1 =
    new G4LogicalVolume(solidShape1,         //its solid
                        shape1_mat,          //its material
                        "Zone_active");           //its name

  new G4PVPlacement(0,                       //no rotation
                    pos1,                    //at position
                    logicShape1,             //its logical volume
                    "Zone_active",                //its name
                    logicEnvelope,                //its mother  volume
                    false,                   //no boolean operation
                    0,                       //copy number
                    checkOverlaps);          //overlaps checking

  //
  // Zone morte
  //
  G4Material* shape2_mat = nist->FindOrBuildMaterial("G4_Ge");
  G4ThreeVector pos2 = G4ThreeVector(0, 0, +2.5*cm);

  G4double pRMin2 = 0*cm;
  G4double pRMax2 = 2*cm;
  G4double pDz2 = 0.5*mm;
  G4double pSPhi2 = 0;
  G4double pDPhi2 = 2*CLHEP::pi;
  G4Tubs* solidShape2 =
      new G4Tubs("Zone_morte",
          pRMin2, pRMax2, pDz2, pSPhi2, pDPhi2);

  G4LogicalVolume* logicShape2 =
      new G4LogicalVolume(solidShape2,         //its solid
          shape2_mat,          //its material
          "Zone_morte");           //its name

  new G4PVPlacement(0,                       //no rotation
      pos2,                    //at position
      logicShape2,             //its logical volume
      "Zone_morte",                //its name
      logicEnvelope,                //its mother  volume
      false,                   //no boolean operation
      0,                       //copy number
      checkOverlaps);          //overlaps checking

  // Set Shape2 as scoring volume
  //
  fScoringVolume = logicShape1;

  //
  //always return the physical World
  //
  return physWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}
