from __future__ import print_function
from sys import stdout
import numpy as np
import routines

"""Evaluate SAPT-FF force field given a PDB.

This is an unmaintained Jesse McDaniel production which may be useful
for doing delta learning on top of a good FF in the future. Here's what
Jesse has to say about this file:

"this script uses OpenMM to evaluate energies of SAPT-FF force field
we use the DrudeSCFIntegrator to solve for equilibrium Drude positions.
positions of atoms are frozen by setting mass = 0 in .xml file"

"""
# datafile with trajectory of coordinates to evaluate force field energy
#inputfile='h2o_h2o_dummy_Eint_10mh.bohr'

# this is used for topology, coordinates are not used...
#pdb = PDBFile('h2o_template.pdb')
def resid(inputfile, pdb):
    # Use the SCF integrator to optimize Drude positions
    integ_md = DrudeSCFIntegrator(0.001 * picoseconds)

    pdb.topology.loadBondDefinitions('water_residue.xml')
    pdb.topology.createStandardBonds()
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField('water_sapt.xml')
    modeller.addExtraParticles(forcefield)

    # by default, no cutoff is used, so all interactions are computed
    system = forcefield.createSystem(
        modeller.topology, constraints=None, rigidWater=True)
    nbondedForce = [
        f for f in [system.getForce(i) for i in range(system.getNumForces())]
        if type(f) == NonbondedForce
    ][0]
    customNonbondedForce = [
        f for f in [system.getForce(i) for i in range(system.getNumForces())]
        if type(f) == CustomNonbondedForce
    ][0]

    for i in range(system.getNumForces()):
        f = system.getForce(i)
        type(f)
        f.setForceGroup(i)

    platform = Platform.getPlatformByName('CPU')
    simmd = Simulation(modeller.topology, system, integ_md, platform)

    #************************* now compute energies from force field *****************************

    # read in dimer configurations
    (aname, xyz, energy) = routines.read_sapt_data2(inputfile)

    # convert coordinates from bohr to nm
    xyz = xyz / 1.88973 / 10.0

    # now loop over data points and compute interaction energies
    #print( ' SAPT energy, Force field energy' )
    eSAPT = np.zeros(len(xyz))
    eFF = np.zeros(len(xyz))
    eResid = np.zeros(len(xyz))
    for i in range(len(xyz)):
        # set coordinates in modeller object,
        # need to use modeller object as this is where initial shell postions are generated
        modeller_pos = routines.set_modeller_coordinates(pdb, modeller, xyz[i])

        # add dummy site and shell initial positions
        modeller_pos.addExtraParticles(forcefield)

        #print( 'step', i )
        #print( modeller_pos.positions)

        # compute energies
        simmd.context.setPositions(modeller_pos.positions)

        # integrate one step to optimize Drude positions
        simmd.step(1)

        state = simmd.context.getState(
            getEnergy=True, getForces=True, getPositions=True)
        # SAPT energy from input file, convert mH to kJ/mol
        eSAPT[i] = energy[i] * 2.6255
        #print(str(state.getPotentialEnergy()).split(" ")[0])
        eFF[i] = float(str(state.getPotentialEnergy()).split(" ")[0])
        eResid[i] = eSAPT[i] - eFF[i]
        #print(eSAPT[i], eFF[i], eResid[i])
        #    for line in en:
        #        col = line.split(line,' ')

        #print( eSAPT , state.getPotentialEnergy())
        #for j in range(system.getNumForces()):
        #    f = system.getForce(j)
        #    print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

    #with open("resid.csv", 'w') as en:
    #    en.write('Resid\n')
    #    for i in range(len(eSAPT)):
    #        en.write('%s \n'%(eResid[i]))
    return (eSAPT, eFF, eResid)


def resid2(inputfile, pdb):
    # Use the SCF integrator to optimize Drude positions
    integ_md = DrudeSCFIntegrator(0.001 * picoseconds)

    pdb.topology.loadBondDefinitions('water_residue.xml')
    pdb.topology.createStandardBonds()
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField('water_sapt.xml')
    modeller.addExtraParticles(forcefield)

    # by default, no cutoff is used, so all interactions are computed
    system = forcefield.createSystem(
        modeller.topology, constraints=None, rigidWater=True)
    nbondedForce = [
        f for f in [system.getForce(i) for i in range(system.getNumForces())]
        if type(f) == NonbondedForce
    ][0]
    customNonbondedForce = [
        f for f in [system.getForce(i) for i in range(system.getNumForces())]
        if type(f) == CustomNonbondedForce
    ][0]

    for i in range(system.getNumForces()):
        f = system.getForce(i)
        type(f)
        f.setForceGroup(i)

    platform = Platform.getPlatformByName('CPU')
    simmd = Simulation(modeller.topology, system, integ_md, platform)

    #************************* now compute energies from force field *****************************

    # read in dimer configurations
    (aname, xyz, energy) = routines.read_sapt_data2(inputfile)

    # convert coordinates from bohr to nm
    xyz = xyz / 1.88973 / 10.0

    # now loop over data points and compute interaction energies
    #print( ' SAPT energy, Force field energy' )
    eSAPT = np.zeros(len(xyz))
    eFF = np.zeros(len(xyz))
    eResid = np.zeros(len(xyz))
    for i in range(len(xyz)):
        # set coordinates in modeller object,
        # need to use modeller object as this is where initial shell postions are generated
        modeller_pos = routines.set_modeller_coordinates(pdb, modeller, xyz[i])

        # add dummy site and shell initial positions
        modeller_pos.addExtraParticles(forcefield)

        #print( 'step', i )
        #print( modeller_pos.positions)

        # compute energies
        simmd.context.setPositions(modeller_pos.positions)

        # integrate one step to optimize Drude positions
        simmd.step(1)

        state = simmd.context.getState(
            getEnergy=True, getForces=True, getPositions=True)
        # SAPT energy from input file, convert mH to kJ/mol
        eSAPT[i] = energy[i] * 2.6255
        #print(str(state.getPotentialEnergy()).split(" ")[0])
        eFF[i] = float(str(state.getPotentialEnergy()).split(" ")[0])
        eResid[i] = eSAPT[i] - eFF[i]
        #print(eSAPT[i], eFF[i], eResid[i])
        #    for line in en:
        #        col = line.split(line,' ')

        #print( eSAPT , state.getPotentialEnergy())
        #for j in range(system.getNumForces()):
        #    f = system.getForce(j)
        #    print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

    #with open("resid.csv", 'w') as en:
    #    en.write('Resid\n')
    #    for i in range(len(eSAPT)):
    #        en.write('%s \n'%(eResid[i]))
    return (eSAPT, eFF, eResid)
