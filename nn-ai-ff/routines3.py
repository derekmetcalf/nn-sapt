from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import sys
import numpy as np
from numpy import linalg as LA
import symmetry_functions as sym
from force_field_parameters import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import losses
import tensorflow as tf

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

def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start,'__getitem__'):
            return arrays[start:stop]
        else:
            return [None]

def create_atype_list(aname, atomic_dict):
    """ 
    creates a lookup list of unique element indices
    """
    index=0
    # fill in first type
    atype=[]
    atypename=[]
    atype.append(index)
    atypename.append(aname[index])  

    for i in range(1, len(aname)):
        # find first match
        flag=-1
        for j in range(i):
            if aname[i] == aname[j]:
               flag=atype[j]
        if flag == -1:
            index+=1
            atype.append(index)
            atypename.append(aname[i])
        else:
            atype.append( flag )
    atomic_number=[]
    for i in range(len(atypename)):
        atomic_number.append(atomic_dict.get(atypename[i],'none'))
    ntype = index + 1
    return ntype, atype, atypename, atomic_number 

def compute_displacements(xyz):
    """
    computes displacement tensor from input cartesian coordinates
    """
    rij = np.zeros((xyz.shape[0],xyz.shape[1],xyz.shape[1])) # tensor of pairwise distances for every data point
    dr = np.zeros((xyz.shape[0],xyz.shape[1],xyz.shape[1],3))
    # loop over data points
    for i in range(len(xyz)):
        # first loop over atoms
        for i_atom in range(len(xyz[0])):
            # second loop over atoms
            for j_atom in range(len(xyz[0])):
                dr[i][i_atom][j_atom] = xyz[i][i_atom] - xyz[i][j_atom]
                rij[i][i_atom][j_atom] = LA.norm(dr[i][i_atom][j_atom])
    return rij,dr

def compute_thetas(rij,dr):
    """
    computes theta tensor from displacement tensor
    """
    theta = np.zeros((rij.shape[0],rij.shape[1],rij.shape[1],rij.shape[1]))
    for i in range(len(rij)):
        for i_atom in range(len(rij[0])):
            for j_atom in range(len(rij[0])):
                for k_atom in range(len(rij[0])):
                    r1 = rij[i][i_atom][j_atom]
                    r2 = rij[i][i_atom][k_atom]
                    dr1 = dr[i][i_atom][j_atom]
                    dr2 = dr[i][i_atom][k_atom]
                    if r1>0 and r2>0 and r1!=r2:
                        theta[i][i_atom][j_atom][k_atom] = np.arccos(np.dot(dr1,dr2)/r1/r2)
                    else:
                        theta[i][i_atom][j_atom][k_atom] = float('nan')
    return theta

def atomwise_displacements( xyz ):
    """
    computes displacement tensor of length 6 vectors from input cartesian coordinates...
    makes rii = 99.9 in order to override cutoff distance and remove self-contribution
    """

    rij=np.zeros( ( xyz.shape[0], xyz.shape[1] , xyz.shape[1] ) ) # tensor of pairwise distances for every data point

    # loop over data points
    for i in range(len(xyz)):
        # first loop over atoms
        for i_atom in range(len(xyz[0])):
            # second loop over atoms
            for j_atom in range(len(xyz[0])):
                if i_atom == j_atom:
                    rij[i][i_atom][j_atom] = 99.9
                else:
                    dr = xyz[i][i_atom] - xyz[i][j_atom]
                    rij[i][i_atom][j_atom] = LA.norm(dr)

    return rij 
      
def flat_displacements( xyz ):
    """
    computes a flattened vector of displacements for every data point
    """
    
    rij = np.zeros((xyz.shape[0],xyz.shape[1]**2))
    for i in range(len(xyz)):
        for i_atom in range(len(xyz[0])):
            for j_atom in range(len(xyz[0])):
                flat_index = j_atom + i_atom*len(xyz[0])
                dr = xyz[i][i_atom] - xyz[i][j_atom]
                rij[i][flat_index] = LA.norm(dr)
    return rij

def construct_symmetry_input( NN , rij, aname , ffenergy, val_split):
    """
    this subroutine constructs input tensor for NN built from
    elemental symmetry functions
 
    NOTE: We assume we are using a shared layer NN structure, where
    layers are shared for common atomtypes (elements).  We then will concatenate
    networks into single layers. For this structure, we need input data to be a list of
    arrays, where each array is the input data (data points, symmetry functions) for each
    atom.

    Thus, the 'sym_input' output of this subroutine is a list of 2D arrays, of form
    [ data_atom1 , data_atom2 , data_atom3 , etc. ] , and data_atomi is an array [ j_data_point , k_symmetry_function ]

    """

    # output tensor with symmetry function input
    sym_input=[]
    split = np.random.binomial(rij.shape[0],val_split)
    print(rij.shape[0])
    # loop over atoms in one system
    for i_atom in range( len(aname) ):
        input_atom=[]
        # get symfun parameters for this atomtype
        typeNN = NN.element_force_field[ aname[i_atom] ]

        # loop over data points
        for i in range( rij.shape[0] ):
            input_i=[ffenergy[i]]
            # now loop over symmetry functions for this element
            for i_sym in range( len( typeNN.symmetry_functions ) ):
                # construct radial gaussian symmetry function
                G1 = sym.radial_gaussian( rij[i] , i_atom , typeNN.symmetry_functions[i_sym][0] , typeNN.symmetry_functions[i_sym][1], typeNN.Rc )
                input_i.append( G1 )

            input_atom.append(input_i)
        # convert 2D list to array
        input_atom = np.array(input_atom)
        sym_input.append(input_atom)
        # append to list
        #if :
        #    sym_input_train.append(input_atom)
        #else:
        #    sym_input_test.append(input_atom)
    return sym_input#sym_input_train, sym_input_test
"""
def construct_symmetry_input( NN , rij, aname , ffenergy, val_split):

    # output tensor with symmetry function input
    sym_input=[]    #list of molecules composed of atoms & associated symfuns
    split = np.random.binomial(rij.shape[0],val_split) #validation/test split
    # loop over molecules in the dataset
    for i in range(rij.shape[0]):
        input_molecule = []     #list of atoms & associated symfuns
        #loop over atoms in the molecule
        for i_atom in range(len(aname)):
            input_atom=[]       #list of associated symfuns
            typeNN = NN.element_force_field[aname[i_atom]]
            #loop over symmetry functions for that atom
            for i_sym in range(len(typeNN.symmetry_functions)):
                G1 = sym.radial_gaussian(rij[i], i_atom,
                            typeNN.symmetry_functions[i_sym][0],
                        typeNN.symmetry_functions[i_sym][1], typeNN.Rc)
                input_atom.append(G1)
            input_atom = np.array(input_atom)
            input_molecule.append(input_atom)
        input_molecule = np.array(input_molecule)
        sym_input.append(input_molecule)

        # append to list
        #if :
        #    sym_input_train.append(input_atom)
        #else:
        #    sym_input_test.append(input_atom)
    return sym_input#sym_input_train, sym_input_test
"""
def construct_flat_symmetry_input( NN , rij, aname , ffenergy):
    """
    This flat version can pretty much only be used when the entire data set has the same # of atoms and atomtypes... so water dimers
    """

    # output tensor with symmetry function input
    flat_sym_input=[]

    # loop over data points
    for i in range( rij.shape[0] ):
        # loop over atoms
        input_i = [ffenergy[i]]
        for i_atom in range( len(aname) ):
            # get elementNN object for this atomtype
            typeNN = NN.element_force_field[ aname[i_atom] ]

            # now loop over symmetry functions for this element
            for i_sym in range( len( typeNN.symmetry_functions ) ):
                # construct radial gaussian symmetry function
                G1 = sym.radial_gaussian( rij[i] , i_atom , typeNN.symmetry_functions[i_sym][0] , typeNN.symmetry_functions[i_sym][1], typeNN.Rc )
                input_i.append( G1 )
        # convert 2D list to array
        input_i = np.array( input_i )
        # append to list
        flat_sym_input.append(input_i)
    flat_sym_input = np.array(flat_sym_input)
    return flat_sym_input

class SymLayer(Layer):
    """
    this is to be used as the first layer of a NN that takes rij and associated atomtypes as input
    """
    def __init__(self,**kwargs):
        super(SymLayer,self).__init__(**kwargs)

    def compute_output_shape(self,input_shape):
        shape = 1   #again, temporary to avoid passing in a bunch of junk for now
        return shape

    def build(self,input_shape):
        self.width_init = K.ones(1)
        self.shift_init = K.ones(1)
        self.width = K.variable(self.width_init,name="width",dtype='float32')
        self.shift = K.variable(self.shift_init,name="shift",dtype='float32')
        self.trainable_weights = [self.width, self.shift]
        super(SymLayer,self).build(input_shape)
        
    def call(self,x):
        Rc = 12.0
        PI = 3.14159265359
        # loop over num. radial sym functions requested
        #for i in range(5):
        # loop over atoms in molec
        G = K.zeros(1,dtype='float32')
        self.width = tf.cast(self.width,tf.float32)
        for j_atom in range( 6 ): #this is super specific for water dimer, fixable
            print(x)
            fc = tf.cond(x<=Rc, lambda: 0.5*(K.cos(PI*x)+1), lambda: 0.0)
            G = G + fc * K.exp(-self.width * (K.pow((x[j_atom] - self.width),2)))
            #K.concatenate([G],axis=0)
        print(G)
        return G


def custom_loss(energy):
    def lossFunction(yTrue,yPred):
        loss = K.mean(K.square(yTrue-yPred))
        loss *= K.exp(-energy)
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

def atomic_dictionary():
    atom_dict = {
        'H'  : 1,
        'He' : 2,
        'Li' : 3,
        'Be' : 4,
        'B' : 5,
        'C' : 6,
        'N' : 7,
        'O' : 8,
        'F' : 9,
        'Ne' : 10,
        'Na' : 11,
        'Mg' : 12,
        'Al' : 13,
        'Si' : 14,
        'P' : 15,
        'S' : 16,
        'Cl' : 17,
        'Ar' :18,
        'K' : 19,
        'Ca' : 20,
        'Sc' : 21,
        'Ti' : 22,
        'V' :  23,
        'Cr' : 24,
        'Mn' : 25,
        'Fe' : 26,
        'Co' : 27,
        'Ni' : 28,
        'Cu' : 29,
        'Zn' : 30,
        'Ga' : 31,
        'Ge' : 32,
        'As' : 33,
        'Se' : 34,
        'Br' : 35,
        'Kr' : 36,
        'Rb' : 37,
        'Sr' : 38,
        'Y' : 39,
        'Zr' : 40,
        'Nb' : 41,
        'Mo' : 42,
        'Tc' : 43,
        'Ru' : 44,
        'Rh' : 45,
        'Pd' : 46,
        'Ag' : 47,
        'Cd' : 48,
        'In' : 49,
        'Sn' : 50,
        'Sb' : 51,
        'Te' : 52,
        'I' : 53,
        'Xe' : 54
    }
    return atom_dict
