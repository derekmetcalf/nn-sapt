from sklearn import linear_model
import numpy as np
import scipy 
from scipy import io
import tensorflow as tf

Z = { 'H' : 1.0,
      'C' : 6.0,
      'N' : 7.0,
      'O' : 8.0,
      'F' : 9.0,
      'S' : 16.0}

def get_Z(elems):
    arr = np.zeros(len(elems))
    for i, e in enumerate(elems):
        arr[i] = Z[e]
    return arr

def get_params():
    params = np.array([[6.3775510, 7.50, 8.00],  
              [6.3775510, 7.22, 8.00],  
              [6.3775510, 6.94, 8.00],  
              [6.3775510, 6.66, 8.00], 
              [6.3775510, 6.38, 8.00], 
              [6.3775510, 6.10, 8.00], 
              [6.3775510, 5.82, 8.00], 
              [6.3775510, 5.54, 8.00], 
              [6.3775510, 5.26, 8.00], 
              [6.3775510, 4.98, 8.00], 
              [6.3775510, 4.70, 8.00], 
              [6.3775510, 4.42, 8.00], 
              [6.3775510, 4.14, 8.00], 
              [6.3775510, 3.86, 8.00], 
              [6.3775510, 3.58, 8.00], 
              [6.3775510, 3.30, 8.00], 
              [6.3775510, 3.02, 8.00], 
              [6.3775510, 2.74, 8.00], 
              [6.3775510, 2.46, 8.00], 
              [6.3775510, 2.18, 8.00], 
              [6.3775510, 1.90, 8.00], 
              [6.3775510, 1.62, 8.00], 
              [6.3775510, 1.34, 8.00], 
              [6.3775510, 1.06, 8.00], 
              [6.3775510, 0.78, 8.00], 
              [6.3775510, 0.50, 8.00]])
    return params

@tf.function
def get_distances(coords):
    return get_distances(coords, coords)

@tf.function
def get_distances_batch(coords1, coords2):
    """ 
    coords1 is B x N x 3
    coords2 is B x M x 3
    N <= M
    returns a N x M Tensor
    """
    N = tf.shape(coords1)[1]
    M = tf.shape(coords2)[1]
    expand1_xyz = tf.expand_dims(coords1, -1) # B x N x 3 x 1
    expand2_xyz = tf.expand_dims(coords2, -1) # B x N x 3 x 1
    tile1_xyz = tf.tile(expand1_xyz,[1,1,1,M]) # B x N x 3 x M
    tile2_xyz = tf.tile(expand2_xyz,[1,1,1,N]) # B x M x 3 x N
    trans2_xyz = tf.transpose(tile2_xyz, perm=[0,3,2,1])
    dR = tf.subtract(tile1_xyz, trans2_xyz)
    #dists = tf.norm(dR, axis=2) # numerically unstable at dR=0
    dists = tf.sqrt(tf.reduce_sum(dR ** 2, axis=2) + 1.0e-8)

    return dists

@tf.function
def get_distances(coords1, coords2):
    """ 
    coords1 is N x 3
    coords2 is M x 3
    N <= M
    returns a N x M Tensor
    """
    N = tf.shape(coords1)[0]
    M = tf.shape(coords2)[0]
    expand1_xyz = tf.expand_dims(coords1, -1) # N x 3 x 1
    expand2_xyz = tf.expand_dims(coords2, -1) # N x 3 x 1
    tile1_xyz = tf.tile(expand1_xyz,[1,1,M]) # N x 3 x M
    tile2_xyz = tf.tile(expand2_xyz,[1,1,N]) # M x 3 x N
    trans2_xyz = tf.transpose(tile2_xyz, perm=[2,1,0])
    dR = tf.subtract(tile1_xyz, trans2_xyz)
    #dists = tf.norm(dR, axis=1) # numerically unstable at dR=0
    dists = tf.sqrt(tf.reduce_sum(dR ** 2, axis=1) + 1.0e-8)

    return dists


@tf.function
def get_cutoff(dists, r_c=8.00):
    r_c_mat = tf.fill(tf.shape(dists), tf.constant(r_c,dtype=tf.float64))
    past_cutoff = tf.add(tf.sign(tf.subtract(r_c_mat, dists)),
                            tf.constant(1.0, dtype=tf.float64))
    cutoff = tf.multiply(tf.constant(0.5, dtype=tf.float64), 
                            tf.add(tf.constant(1.0, dtype=tf.float64),
                            tf.cos(tf.divide(tf.multiply(
                            tf.constant(np.pi, dtype=tf.float64), dists),
                            r_c))))
    cutoff = tf.multiply(tf.multiply(cutoff, past_cutoff),
                            tf.constant(0.5, dtype=tf.float64))
    return cutoff

@tf.function
def get_rsym_batch(Z, distances, rparams):

    B = tf.shape(distances)[0]
    N = tf.shape(distances)[1]
    M = tf.shape(distances)[2]
    R = tf.shape(rparams)[0]

    cutoff = get_cutoff(distances) # B x N x M
    cutoff = tf.expand_dims(cutoff, 1) # B x 1 x N x M
    cutoff = tf.tile(cutoff, [1, R, 1, 1]) # B x R x N x M
    
    weights = tf.expand_dims(Z, 1) # B x 1 x M
    weights = tf.tile(weights,[1, N, 1]) # B x N x M
    weights = tf.expand_dims(weights, 1) # B x 1 x N x M
    weights = tf.tile(weights,[1,R,1,1]) # B x R x N x M
    
    expand_dist = tf.expand_dims(distances, 1) # B x 1 x N x M
    tile_dist = tf.tile(expand_dist, [1,R,1,1]) # B x R x N x M

    eta, mu, rc_ph = tf.unstack(rparams, axis=1) # R

    eta = tf.expand_dims(eta, -1)
    eta = tf.tile(eta, [1,N])
    eta = tf.expand_dims(eta, -1)
    eta = tf.tile(eta, [1,1,M])
    eta = tf.expand_dims(eta, 0)
    eta = tf.tile(eta, [B,1,1,1]) # B x R x N x M

    mu = tf.expand_dims(mu, -1)
    mu = tf.tile(mu, [1,N])
    mu = tf.expand_dims(mu, -1)
    mu = tf.tile(mu, [1,1,M])
    mu = tf.expand_dims(mu, 0)
    mu = tf.tile(mu, [B,1,1,1]) # B x R x N x M
   
    symfuns = tf.multiply(weights, tf.multiply(tf.exp(tf.multiply(-eta,tf.square(tf.subtract(tile_dist, mu)))), cutoff))
    symfuns = tf.reduce_sum(symfuns, axis=3)
    symfuns = tf.transpose(symfuns, perm=[0,2,1])

    return symfuns

@tf.function
def get_rsym(Z, distances, rparams):

    N = tf.shape(distances)[0]
    M = tf.shape(distances)[1]
    R = tf.shape(rparams)[0]

    cutoff = get_cutoff(distances) # N x M
    cutoff = tf.expand_dims(cutoff, 0) # 1 x N x M
    cutoff = tf.tile(cutoff, [R, 1, 1]) # R x N x M
    
    weights = tf.expand_dims(Z, 0) # 1 x M
    weights = tf.tile(weights,[N, 1]) # N x M
    weights = tf.expand_dims(weights, 0) # 1 x N x M
    weights = tf.tile(weights,[R,1,1]) # R x N x M
    
    expand_dist = tf.expand_dims(distances, 0) # 1 x N x M
    tile_dist = tf.tile(expand_dist, [R,1,1]) # R x N x M

    eta, mu, rc_ph = tf.unstack(rparams, axis=1) # R

    eta = tf.expand_dims(eta, -1)
    eta = tf.tile(eta, [1,N])
    eta = tf.expand_dims(eta, -1)
    eta = tf.tile(eta, [1,1,M]) # R x N x M

    mu = tf.expand_dims(mu, -1)
    mu = tf.tile(mu, [1,N])
    mu = tf.expand_dims(mu, -1)
    mu = tf.tile(mu, [1,1,M]) # R x N x M
   
    symfuns = tf.multiply(weights, tf.multiply(tf.exp(tf.multiply(-eta,tf.square(tf.subtract(tile_dist, mu)))), cutoff))
    symfuns = tf.reduce_sum(symfuns, axis=2)
    symfuns = tf.transpose(symfuns)

    return symfuns

def get_qm9(size=None):

    xyz = np.load('./data/qm9/xyz.npy') # in angstrom
    Z = np.load('./data/qm9/Z.npy')
    labels = np.load('./data/qm9/labels.npy') * 627.509 # Ha to kcal/mol
    
    if size is not None:
        xyz = xyz[:size]
        Z = Z[:size]
        labels = labels[:size]

    atom_Z = [1.0, 6.0, 7.0, 8.0, 9.0]
    return get_dataset(xyz, Z, labels, atom_Z)


def get_qm7(size=None):

    data = scipy.io.loadmat('./data/qm7.mat')
    coords_all = data['R'] * 0.529177249 # bohr to angstrom
    Z = data['Z'] 
    labels = data['T'][0] # already in kcal / mol
    
    if size is not None:
        coords_all = coords_all[:size]
        Z = Z[:size]
        labels = labels[:size]

    atom_Z = [1.0, 6.0, 7.0, 8.0, 16.0]
    return get_dataset(coords_all, Z, labels, atom_Z)

def get_dataset(coords_all, Z, labels, atom_Z):

    M = len(labels)
    print('Loading QM7')

    max_atoms = []

    atom_counts = np.zeros((M,len(atom_Z)))
    for i, z in enumerate(atom_Z):

        is_atom = (Z == z)
        atoms_per_mol = np.sum(is_atom, axis=1)
        atom_counts[:,i] = atoms_per_mol
        max_atoms.append(np.max(atoms_per_mol))

    regr = linear_model.LinearRegression()
    regr.fit(atom_counts, labels)
    labels_pred = regr.predict(atom_counts)
    coeffs = regr.coef_
    print('Linear Regression Coefficients')
    for i, z in enumerate(atom_Z):
        print(f'  Z={z} : {coeffs[i]}')

    delta = labels - labels_pred
    print(f'Label Statistics: Average : {np.average(labels):.2f} +- {np.std(labels):.2f}')
    print(f'                  Range :   [{np.min(labels):.2f}, {np.max(labels):.2f}]' )
    print(f'Delta Statistics: Average : {np.average(delta):.2f} +- {np.std(delta):.2f}' )
    print(f'                  Range :   [{np.min(delta):.2f}, {np.max(delta):.2f}]' )

    ### CONSIDER COMMENTING THIS LINE OUT
    labels = delta

    coords_atom = []
    masks_atom = []
    for zi, pad in enumerate(max_atoms):
        
        z = atom_Z[zi]
        coords = np.full((M,pad,3), 900.0)
        is_atom = (Z == z)


        for mi, molecule in enumerate(coords_all):
            spot = 0
            for ai, atom_coord in enumerate(molecule):
                if is_atom[mi,ai]:
                    coords[mi,spot] = atom_coord
                    spot += 1
        is_atom2 = (coords[:,:,0] != 900.0)
        mask_i = np.zeros_like(is_atom2, dtype=np.float)
        mask_i[is_atom2] = 1.0

        coords_atom.append(tf.convert_to_tensor(coords, dtype=tf.float64))
        masks_atom.append(tf.convert_to_tensor(mask_i, dtype=tf.float64))

    is_negative_ghost = (Z == 0.0)
    coords_all[is_negative_ghost] = np.array([-900.0, -900.0, -900.0])
    
    coords_all = tf.convert_to_tensor(coords_all, dtype=tf.float64)
    Z = tf.convert_to_tensor(Z, dtype=tf.float64)
    labels = tf.convert_to_tensor(labels, dtype=tf.float64)
    
    return coords_atom, masks_atom, coords_all, Z, labels, atom_Z, coeffs


if __name__ == "__main__":

    xyz_l, mask_l, xyz_all, Z, labels = get_qm7()

