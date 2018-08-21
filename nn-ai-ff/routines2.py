from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import sys
import numpy as np
from numpy import linalg as LA
import symmetry_functions as sym
from force_field_parameters import *
from keras import backend as K
from keras import losses

"""
these are general subroutines that we use for the NN program
these subroutines are meant to be indepedent of tensorflow/keras libraries,
and are used for I/O, data manipulation (construction of symmetry functions), etc.
"""

def read_sapt_data2( inputfile ):
      """
      reads SAPT interaction energy input file, outputs
      list of all atomic coordinates, and total interaction energies
      """

      file = open(inputfile)
      aname=[]
      xyz=[]
      energy=[]
       
      for line in file:
          # atoms first molecule
          atoms=int(line.rstrip())
          # empty list for this dimer
          dimeraname=[]
          dimerxyz=[]
          dimerxyzD=[]
          # first molecule
          for i in range(atoms):
              line=file.readline()
              data=line.split()
              # add xyz to dimer list
              dimeraname.append( data[0] )
              dimerxyz.append( [ float(data[1]) , float(data[2]) , float(data[3]) ] )
          # pop dummy atom off list
          dimeraname.pop()
          dimerxyz.pop()
          # second molecule
          line=file.readline()
          atoms=int(line.rstrip())
          for i in range(atoms):
              line=file.readline()
              data=line.split()
              # add xyz to dimer list
              dimeraname.append( data[0] )
              dimerxyz.append( [ float(data[1]) , float(data[2]) , float(data[3]) ] )
          # pop dummy atom off list
          dimeraname.pop()
          dimerxyz.pop()

          aname.append( dimeraname )
          xyz.append( dimerxyz )
          # now read energy SAPT terms
          for i in range(19):
              line=file.readline()
              # get E1tot+E2tot
              if i == 12 :
                  data = line.split()
                  e1tote2tot= float(data[1])
              # get delta HF energy
              elif i == 17 :
                  data = line.split()
                  dhf= float(data[1])                  
              # save total energy
          etot = e1tote2tot + dhf
          energy.append( etot )

      # change lists to arrays
      xyz=np.array(xyz)
      energy=np.array(energy)

      dimeraname = aname[0]

      return dimeraname, xyz, energy



def set_modeller_coordinates( pdb , modeller , xyz_in ):
    """
    this sets the coordinates of the OpenMM 'pdb' and 'modeller' objects
    with the input coordinate array 'xyz_in'
    """
    # loop over atoms
    for i in range(len(xyz_in)):
        # update pdb positions
        pdb.positions[i]=Vec3(xyz_in[i][0] , xyz_in[i][1], xyz_in[i][2])*nanometer

    # now update positions in modeller object
    modeller_out = Modeller(pdb.topology, pdb.positions)
  
    return modeller_out



def create_atype_list( aname ):
    """ 
    creates a lookup list of unique element indices
    """
    index=0
    # fill in first type
    atype=[]
    atypename=[]
    atype.append( index )
    atypename.append( aname[index] )  

    for i in range(1, len(aname)):
        # find first match
        flag=-1
        for j in range(i):
            if aname[i] == aname[j]:
               flag=atype[j]
        if flag == -1:
            index+=1
            atype.append( index )
            atypename.append( aname[i] )
        else:
            atype.append( flag )

    ntype = index + 1
    return ntype, atype, atypename 



def compute_displacements( xyz ):
    """
    computes displacement tensor from input cartesian coordinates
    """

    rij=np.zeros( ( xyz.shape[0] , xyz.shape[1], xyz.shape[1] ) ) # tensor of pairwise distances for every data point

    # loop over data points
    for i in range(len(xyz)):
        # first loop over atoms
        for i_atom in range(len(xyz[0])):
            # second loop over atoms
            for j_atom in range(len(xyz[0])):
                dr = xyz[i][i_atom] - xyz[i][j_atom]
                rij[i][i_atom][j_atom] = LA.norm(dr)

    return rij 
      

def construct_symmetry_input( NN , rij, aname ):
    """
    this subroutine constructs input tensor for NN built from
    elemental symmetry functions
 
    NOTE: We assume we are using a shared layer NN structure, where
    layers are shared for common atomtypes (elements).  We then will concatenate
    nodes into single layers.  For this structure, we need input data to be a list of
    arrays, where each array is the input data (data points, symmetry functions) for each
    atom.  

    Thus, the 'sym_input' output of this subroutine is a list of 2D arrays, of form
    [ data_atom1 , data_atom2 , data_atom3 , etc. ] , and data_atomi is an array [ j_data_point , k_symmetry_function ]

    """

    # output tensor with symmetry function input
    sym_input=[]


    # loop over atoms
    for i_atom in range( len(aname) ):
        input_atom=[]
        # get elementNN object for this atomtype
        typeNN = NN.element_force_field[ aname[i_atom] ]

        # loop over data points
        for i in range( rij.shape[0] ):
            input_i=[]

            # now loop over symmetry functions for this element
            for i_sym in range( len( typeNN.symmetry_functions ) ):
                # construct radial gaussian symmetry function
                G1 = sym.radial_gaussian( rij[i] , i_atom , typeNN.symmetry_functions[i_sym][0] , typeNN.symmetry_functions[i_sym][1], typeNN.Rc )
                input_i.append( G1 )

            input_atom.append( input_i ) 

        # convert 2D list to array
        input_atom = np.array( input_atom )
        # append to list
        sym_input.append( input_atom )

    return sym_input 


def custom_loss(en):
    def lossFunction(y_true,y_pred):
        loss = losses.mean_squared_error(y_true,y_pred)
        loss *= K.exp(-en)
    return lossFunction

def scale_symmetry_input( sym_input ):
    """
    this subroutine scales the symmetry input to lie within the range [-1,1]
    this helps improve the NN performance, but makes the NN and input data set dependent!
    """
    small=1E-6
    # loop over atoms
    for i_atom in range( len(sym_input) ):
        # loop over symmetry functions
        for i_sym in range( sym_input[i_atom].shape[1] ):
            # min and max values of this symmetry function across the data set
            min_sym = np.amin(sym_input[i_atom][:,i_sym] )
            max_sym = np.amax(sym_input[i_atom][:,i_sym] )
            range_sym = max_sym - min_sym

            if range_sym > small:
                for i_data in range( sym_input[i_atom].shape[0] ):
                    sym_input[i_atom][i_data,i_sym] = -1.0 + 2.0 * ( ( sym_input[i_atom][i_data,i_sym] - min_sym ) / range_sym )
            
            # shift symmetry function value so that histogram maximum is centered at 0
            # this is for increasing sensitivity to tanh activation function
            hist = np.histogram( sym_input[i_atom][:,i_sym] , bins=20 )
            shift = np.argmax(hist[0])
            array_shift = np.full_like( sym_input[i_atom][:,i_sym] , -1.0 + 2*( shift / 20 ) )
            sym_input[i_atom][:,i_sym] = sym_input[i_atom][:,i_sym] - array_shift[:]
            

    return sym_input



def test_symmetry_input(sym_input):
    """
    for a symmetry function to contribute to a neural network
    it must exhibit a non-negligable variance across the data set
    here we test the variance of symmetry functions across data set
    """

    # loop over atoms
    for i_atom in range( len(sym_input) ):
        print ("symmetry functions for atom ", i_atom )
        
        # histogram values of every symmetry function
        for i_sym in range( sym_input[i_atom].shape[1] ):
  
            hist = np.histogram( sym_input[i_atom][:,i_sym] )

            for i in range(len(hist[0])):
                print( hist[1][i] , hist[0][i] )
            print( "" )

    sys.exit()















 






















   








