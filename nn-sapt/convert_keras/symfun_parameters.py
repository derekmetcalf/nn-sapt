import sys
import numpy as np
"""Define elemental parameters that are used to construct symmetry functions.

Defines classes of symmetry function parameters with each class corresponding
to a different collection of symmetry function types and definitions.
"""


class elementNN(object):
    """Store all definitions for element-specific neural network."""

    def __init__(self):
        """Initialize data structures."""
        self.radial_symmetry_functions = []
        self.angular_symmetry_functions = []
        self.weights = []
        # set default cutoff radius (Rc) for Gaussian symmetry functions
        self.Rc = 11.5  # Bohr


class NNforce_field(object):
    """Define symmetry function parameter object

    Stores parameters that define the atomic symmetry functions.
    
    HISTORICAL NOTE:
    These have nothing to do with force fields in the MM sense.
    Jesse McDaniel corresponds symmetry function parameters to 
    force field parameters, since they and the model weights ultimately
    determine preditcions of a system much like FF parameters. Since 
    Dr. McDaniel wrote an early version of this code, this name has 
    been embedded deeply in the lore of NN-SAPT and would frankly 
    be hard to take out at this point.
    """

    def __init__(self, name, i, j):
        """Setup attributes of symmetry functions based on name."""
        if name is 'FF':
            self.initialize_FF(i, j)
        elif name is 'FF1':
            self.initialize_FF1()
        elif name is 'FF2':
            self.initialize_FF2()
        elif name is 'FF3':
            self.initialize_FF3()
        elif name is 'GA_opt':
            self.initialize_GA_opt()
        else:
            print("symmetry funciton name ", name, "  is not recognized")
            sys.exit()


    def initialize_FF(self, num_widths, num_cutoffs):
        """Construct symfuns based on a naive range-based parametrization."""

        # create a dictionary to look up force field object for each element
        self.element_force_field = {}

        # for now this is just a list of tuples that define the Gaussian width
        # shifts, and cutoff value for each Gaussian symmetry function.  Eventually, this will be
        # a more complicated object...

        gauss_width_min = 0.5
        gauss_width_max = 2.0

        cutoff_min = 3.0
        cutoff_max = 6.0

        gauss_vec = np.linspace(gauss_width_min, gauss_width_max, num_widths)
        cutoff_vec = np.linspace(cutoff_min, cutoff_max, num_cutoffs)
        print(range(num_widths))
        # setup force field for oxygen
        oxygenNN = elementNN()
        self.element_force_field["O"] = oxygenNN
        # add list of symmetry function tuples
        for i in range(num_widths):
            for j in range(num_cutoffs):
                oxygenNN.symmetry_functions.append((gauss_vec[i],
                                                    cutoff_vec[j]))

        # setup force field for hydrogen
        hydrogenNN = elementNN()
        self.element_force_field["H"] = hydrogenNN
        # add list of symmetry function tuples
        for i in range(num_widths):
            for j in range(num_cutoffs):
                hydrogenNN.symmetry_functions.append((gauss_vec[i],
                                                      cutoff_vec[j]))

    def initialize_FF1(self):
        """Initialize first collection of symmetry functions."""

        # create a dictionary to look up force field object for each element
        self.element_force_field = {}

        # for now this is just a list of tuples that define the Gaussian width
        # shifts, and cutoff value for each Gaussian symmetry function.  Eventually, this will be
        # a more complicated object...

        # setup force field for oxygen
        oxygenNN = elementNN()
        self.element_force_field["O"] = oxygenNN
        # add list of symmetry function tuples
        oxygenNN.symmetry_functions.append((3.0, 3.0))
        oxygenNN.symmetry_functions.append((3.0, 3.5))
        oxygenNN.symmetry_functions.append((3.0, 4.0))
        oxygenNN.symmetry_functions.append((3.0, 4.5))
        oxygenNN.symmetry_functions.append((3.0, 5.0))
        oxygenNN.symmetry_functions.append((3.0, 5.5))
        oxygenNN.symmetry_functions.append((3.0, 6.0))
        oxygenNN.symmetry_functions.append((3.0, 6.5))
        oxygenNN.symmetry_functions.append((3.0, 7.0))

        # setup force field for hydrogen
        hydrogenNN = elementNN()
        self.element_force_field["H"] = hydrogenNN
        # add list of symmetry function tuples

        hydrogenNN.symmetry_functions.append((3.0, 3.0))
        hydrogenNN.symmetry_functions.append((3.0, 3.5))
        hydrogenNN.symmetry_functions.append((3.0, 4.0))
        hydrogenNN.symmetry_functions.append((3.0, 4.5))
        hydrogenNN.symmetry_functions.append((3.0, 5.0))
        hydrogenNN.symmetry_functions.append((3.0, 5.5))
        hydrogenNN.symmetry_functions.append((3.0, 6.0))
        hydrogenNN.symmetry_functions.append((3.0, 6.5))
        hydrogenNN.symmetry_functions.append((3.0, 7.0))

    def initialize_FF2(self):
        # create a dictionary to look up force field object for each element
        self.element_force_field = {}

        # for now this is just a list of tuples that define the Gaussian width
        # shifts, and cutoff value for each Gaussian symmetry function.  Eventually, this will be
        # a more complicated object...

        # setup force field for oxygen
        oxygenNN = elementNN()
        self.element_force_field["O"] = oxygenNN
        # add list of symmetry function tuples
        oxygenNN.radial_symmetry_functions.append((3.0, 3.0))
        oxygenNN.radial_symmetry_functions.append((3.0, 3.333))
        oxygenNN.radial_symmetry_functions.append((3.0, 3.667))
        oxygenNN.radial_symmetry_functions.append((3.0, 4.0))
        oxygenNN.radial_symmetry_functions.append((3.0, 4.333))
        oxygenNN.radial_symmetry_functions.append((3.0, 4.667))
        oxygenNN.radial_symmetry_functions.append((3.0, 5.0))
        oxygenNN.radial_symmetry_functions.append((3.0, 5.333))
        oxygenNN.radial_symmetry_functions.append((3.0, 5.667))
        oxygenNN.radial_symmetry_functions.append((3.0, 6.0))
        oxygenNN.radial_symmetry_functions.append((3.0, 6.333))
        oxygenNN.radial_symmetry_functions.append((3.0, 6.667))
        oxygenNN.radial_symmetry_functions.append((3.0, 7.0))
        oxygenNN.radial_symmetry_functions.append((3.0, 7.333))
        oxygenNN.radial_symmetry_functions.append((3.0, 7.667))
        oxygenNN.radial_symmetry_functions.append((4.0, 3.0))
        oxygenNN.radial_symmetry_functions.append((4.0, 3.333))
        oxygenNN.radial_symmetry_functions.append((4.0, 3.667))
        oxygenNN.radial_symmetry_functions.append((4.0, 4.0))
        oxygenNN.radial_symmetry_functions.append((4.0, 4.333))
        oxygenNN.radial_symmetry_functions.append((4.0, 4.667))
        oxygenNN.radial_symmetry_functions.append((4.0, 5.0))
        oxygenNN.radial_symmetry_functions.append((4.0, 5.333))
        oxygenNN.radial_symmetry_functions.append((4.0, 5.667))
        oxygenNN.radial_symmetry_functions.append((4.0, 6.0))
        oxygenNN.radial_symmetry_functions.append((4.0, 6.333))
        oxygenNN.radial_symmetry_functions.append((4.0, 6.667))
        oxygenNN.radial_symmetry_functions.append((4.0, 7.0))
        oxygenNN.radial_symmetry_functions.append((4.0, 7.333))
        oxygenNN.radial_symmetry_functions.append((4.0, 7.667))
        oxygenNN.radial_symmetry_functions.append((2.0, 3.0))
        oxygenNN.radial_symmetry_functions.append((2.0, 3.333))
        oxygenNN.radial_symmetry_functions.append((2.0, 3.667))
        oxygenNN.radial_symmetry_functions.append((2.0, 4.0))
        oxygenNN.radial_symmetry_functions.append((2.0, 4.333))
        oxygenNN.radial_symmetry_functions.append((2.0, 4.667))
        oxygenNN.radial_symmetry_functions.append((2.0, 5.0))
        oxygenNN.radial_symmetry_functions.append((2.0, 5.333))
        oxygenNN.radial_symmetry_functions.append((2.0, 5.667))
        oxygenNN.radial_symmetry_functions.append((2.0, 6.0))
        oxygenNN.radial_symmetry_functions.append((2.0, 6.333))
        oxygenNN.radial_symmetry_functions.append((2.0, 6.667))
        oxygenNN.radial_symmetry_functions.append((2.0, 7.0))
        oxygenNN.radial_symmetry_functions.append((2.0, 7.333))
        oxygenNN.radial_symmetry_functions.append((2.0, 3.667))

        oxygenNN.angular_symmetry_functions.append((3.0, 3.0, -1))
        oxygenNN.angular_symmetry_functions.append((3.0, 3.5, -1))
        oxygenNN.angular_symmetry_functions.append((3.0, 4.0, -1))
        oxygenNN.angular_symmetry_functions.append((3.0, 4.5, 1))
        oxygenNN.angular_symmetry_functions.append((3.0, 5.0, 1))
        oxygenNN.angular_symmetry_functions.append((3.0, 5.5, 1))
        oxygenNN.angular_symmetry_functions.append((4.0, 3.0, -1))
        oxygenNN.angular_symmetry_functions.append((4.0, 3.5, -1))
        oxygenNN.angular_symmetry_functions.append((4.0, 4.0, -1))
        oxygenNN.angular_symmetry_functions.append((4.0, 4.5, 1))
        oxygenNN.angular_symmetry_functions.append((4.0, 5.0, 1))
        oxygenNN.angular_symmetry_functions.append((4.0, 5.5, 1))

        # setup force field for hydrogen
        hydrogenNN = elementNN()
        self.element_force_field["H"] = hydrogenNN
        # add list of symmetry function tuples
        hydrogenNN.radial_symmetry_functions.append((2.0, 3.0))
        hydrogenNN.radial_symmetry_functions.append((2.0, 3.5))
        hydrogenNN.radial_symmetry_functions.append((2.0, 4.0))
        hydrogenNN.radial_symmetry_functions.append((2.0, 4.5))
        hydrogenNN.radial_symmetry_functions.append((2.0, 5.0))
        hydrogenNN.radial_symmetry_functions.append((2.0, 5.5))
        hydrogenNN.radial_symmetry_functions.append((2.0, 6.0))
        hydrogenNN.radial_symmetry_functions.append((2.0, 6.5))
        hydrogenNN.radial_symmetry_functions.append((2.0, 7.0))
        hydrogenNN.radial_symmetry_functions.append((3.0, 3.0))
        hydrogenNN.radial_symmetry_functions.append((3.0, 3.5))
        hydrogenNN.radial_symmetry_functions.append((3.0, 4.0))
        hydrogenNN.radial_symmetry_functions.append((3.0, 4.5))
        hydrogenNN.radial_symmetry_functions.append((3.0, 5.0))
        hydrogenNN.radial_symmetry_functions.append((3.0, 5.5))
        hydrogenNN.radial_symmetry_functions.append((3.0, 6.0))
        hydrogenNN.radial_symmetry_functions.append((3.0, 6.5))
        hydrogenNN.radial_symmetry_functions.append((3.0, 7.0))
        hydrogenNN.radial_symmetry_functions.append((4.0, 3.0))
        hydrogenNN.radial_symmetry_functions.append((4.0, 3.5))
        hydrogenNN.radial_symmetry_functions.append((4.0, 4.0))
        hydrogenNN.radial_symmetry_functions.append((4.0, 4.5))
        hydrogenNN.radial_symmetry_functions.append((4.0, 5.0))
        hydrogenNN.radial_symmetry_functions.append((4.0, 5.5))
        hydrogenNN.radial_symmetry_functions.append((4.0, 6.0))
        hydrogenNN.radial_symmetry_functions.append((4.0, 6.5))
        hydrogenNN.radial_symmetry_functions.append((4.0, 7.0))
        hydrogenNN.angular_symmetry_functions.append((3.0, 3.0, -1))
        hydrogenNN.angular_symmetry_functions.append((3.0, 3.5, -1))
        hydrogenNN.angular_symmetry_functions.append((3.0, 4.0, -1))
        hydrogenNN.angular_symmetry_functions.append((3.0, 4.5, 1))
        hydrogenNN.angular_symmetry_functions.append((3.0, 5.0, 1))
        hydrogenNN.angular_symmetry_functions.append((3.0, 5.5, 1))

        carbonNN = elementNN()
        self.element_force_field["C"] = carbonNN
        # add list of symmetry function tuples

        carbonNN.radial_symmetry_functions.append((3.0, 3.0))
        carbonNN.radial_symmetry_functions.append((3.0, 3.333))
        carbonNN.radial_symmetry_functions.append((3.0, 3.667))
        carbonNN.radial_symmetry_functions.append((3.0, 4.0))
        carbonNN.radial_symmetry_functions.append((3.0, 4.333))
        carbonNN.radial_symmetry_functions.append((3.0, 4.667))
        carbonNN.radial_symmetry_functions.append((3.0, 5.0))
        carbonNN.radial_symmetry_functions.append((3.0, 5.333))
        carbonNN.radial_symmetry_functions.append((3.0, 5.667))
        carbonNN.radial_symmetry_functions.append((3.0, 6.0))
        carbonNN.radial_symmetry_functions.append((3.0, 6.333))
        carbonNN.radial_symmetry_functions.append((3.0, 6.667))
        carbonNN.radial_symmetry_functions.append((3.0, 7.0))
        carbonNN.radial_symmetry_functions.append((3.0, 7.333))
        carbonNN.radial_symmetry_functions.append((3.0, 7.667))
        carbonNN.radial_symmetry_functions.append((4.0, 3.0))
        carbonNN.radial_symmetry_functions.append((4.0, 3.333))
        carbonNN.radial_symmetry_functions.append((4.0, 3.667))
        carbonNN.radial_symmetry_functions.append((4.0, 4.0))
        carbonNN.radial_symmetry_functions.append((4.0, 4.333))
        carbonNN.radial_symmetry_functions.append((4.0, 4.667))
        carbonNN.radial_symmetry_functions.append((4.0, 5.0))
        carbonNN.radial_symmetry_functions.append((4.0, 5.333))
        carbonNN.radial_symmetry_functions.append((4.0, 5.667))
        carbonNN.radial_symmetry_functions.append((4.0, 6.0))
        carbonNN.radial_symmetry_functions.append((4.0, 6.333))
        carbonNN.radial_symmetry_functions.append((4.0, 6.667))
        carbonNN.radial_symmetry_functions.append((4.0, 7.0))
        carbonNN.radial_symmetry_functions.append((4.0, 7.333))
        carbonNN.radial_symmetry_functions.append((4.0, 7.667))
        carbonNN.radial_symmetry_functions.append((2.0, 3.0))
        carbonNN.radial_symmetry_functions.append((2.0, 3.333))
        carbonNN.radial_symmetry_functions.append((2.0, 3.667))
        carbonNN.radial_symmetry_functions.append((2.0, 4.0))
        carbonNN.radial_symmetry_functions.append((2.0, 4.333))
        carbonNN.radial_symmetry_functions.append((2.0, 4.667))
        carbonNN.radial_symmetry_functions.append((2.0, 5.0))
        carbonNN.radial_symmetry_functions.append((2.0, 5.333))
        carbonNN.radial_symmetry_functions.append((2.0, 5.667))
        carbonNN.radial_symmetry_functions.append((2.0, 6.0))
        carbonNN.radial_symmetry_functions.append((2.0, 6.333))
        carbonNN.radial_symmetry_functions.append((2.0, 6.667))
        carbonNN.radial_symmetry_functions.append((2.0, 7.0))
        carbonNN.radial_symmetry_functions.append((2.0, 7.333))
        carbonNN.radial_symmetry_functions.append((2.0, 3.667))
        carbonNN.angular_symmetry_functions.append((3.0, 3.0, -1))
        carbonNN.angular_symmetry_functions.append((3.0, 3.5, -1))
        carbonNN.angular_symmetry_functions.append((3.0, 4.0, -1))
        carbonNN.angular_symmetry_functions.append((3.0, 4.5, 1))
        carbonNN.angular_symmetry_functions.append((3.0, 5.0, 1))
        carbonNN.angular_symmetry_functions.append((3.0, 5.5, 1))
        carbonNN.angular_symmetry_functions.append((4.0, 3.0, -1))
        carbonNN.angular_symmetry_functions.append((4.0, 3.5, -1))
        carbonNN.angular_symmetry_functions.append((4.0, 4.0, -1))
        carbonNN.angular_symmetry_functions.append((4.0, 4.5, 1))
        carbonNN.angular_symmetry_functions.append((4.0, 5.0, 1))
        carbonNN.angular_symmetry_functions.append((4.0, 5.5, 1))

        nitrogenNN = elementNN()
        self.element_force_field["N"] = oxygenNN
        # add list of symmetry function tuples
        nitrogenNN.radial_symmetry_functions.append((3.0, 3.0))
        nitrogenNN.radial_symmetry_functions.append((3.0, 3.333))
        nitrogenNN.radial_symmetry_functions.append((3.0, 3.667))
        nitrogenNN.radial_symmetry_functions.append((3.0, 4.0))
        nitrogenNN.radial_symmetry_functions.append((3.0, 4.333))
        nitrogenNN.radial_symmetry_functions.append((3.0, 4.667))
        nitrogenNN.radial_symmetry_functions.append((3.0, 5.0))
        nitrogenNN.radial_symmetry_functions.append((3.0, 5.333))
        nitrogenNN.radial_symmetry_functions.append((3.0, 5.667))
        nitrogenNN.radial_symmetry_functions.append((3.0, 6.0))
        nitrogenNN.radial_symmetry_functions.append((3.0, 6.333))
        nitrogenNN.radial_symmetry_functions.append((3.0, 6.667))
        nitrogenNN.radial_symmetry_functions.append((3.0, 7.0))
        nitrogenNN.radial_symmetry_functions.append((3.0, 7.333))
        nitrogenNN.radial_symmetry_functions.append((3.0, 7.667))
        nitrogenNN.radial_symmetry_functions.append((4.0, 3.0))
        nitrogenNN.radial_symmetry_functions.append((4.0, 3.333))
        nitrogenNN.radial_symmetry_functions.append((4.0, 3.667))
        nitrogenNN.radial_symmetry_functions.append((4.0, 4.0))
        nitrogenNN.radial_symmetry_functions.append((4.0, 4.333))
        nitrogenNN.radial_symmetry_functions.append((4.0, 4.667))
        nitrogenNN.radial_symmetry_functions.append((4.0, 5.0))
        nitrogenNN.radial_symmetry_functions.append((4.0, 5.333))
        nitrogenNN.radial_symmetry_functions.append((4.0, 5.667))
        nitrogenNN.radial_symmetry_functions.append((4.0, 6.0))
        nitrogenNN.radial_symmetry_functions.append((4.0, 6.333))
        nitrogenNN.radial_symmetry_functions.append((4.0, 6.667))
        nitrogenNN.radial_symmetry_functions.append((4.0, 7.0))
        nitrogenNN.radial_symmetry_functions.append((4.0, 7.333))
        nitrogenNN.radial_symmetry_functions.append((4.0, 7.667))
        nitrogenNN.radial_symmetry_functions.append((2.0, 3.0))
        nitrogenNN.radial_symmetry_functions.append((2.0, 3.333))
        nitrogenNN.radial_symmetry_functions.append((2.0, 3.667))
        nitrogenNN.radial_symmetry_functions.append((2.0, 4.0))
        nitrogenNN.radial_symmetry_functions.append((2.0, 4.333))
        nitrogenNN.radial_symmetry_functions.append((2.0, 4.667))
        nitrogenNN.radial_symmetry_functions.append((2.0, 5.0))
        nitrogenNN.radial_symmetry_functions.append((2.0, 5.333))
        nitrogenNN.radial_symmetry_functions.append((2.0, 5.667))
        nitrogenNN.radial_symmetry_functions.append((2.0, 6.0))
        nitrogenNN.radial_symmetry_functions.append((2.0, 6.333))
        nitrogenNN.radial_symmetry_functions.append((2.0, 6.667))
        nitrogenNN.radial_symmetry_functions.append((2.0, 7.0))
        nitrogenNN.radial_symmetry_functions.append((2.0, 7.333))
        nitrogenNN.radial_symmetry_functions.append((2.0, 3.667))

        nitrogenNN.angular_symmetry_functions.append((3.0, 3.0, -1))
        nitrogenNN.angular_symmetry_functions.append((3.0, 3.5, -1))
        nitrogenNN.angular_symmetry_functions.append((3.0, 4.0, -1))
        nitrogenNN.angular_symmetry_functions.append((3.0, 4.5, 1))
        nitrogenNN.angular_symmetry_functions.append((3.0, 5.0, 1))
        nitrogenNN.angular_symmetry_functions.append((3.0, 5.5, 1))
        nitrogenNN.angular_symmetry_functions.append((4.0, 3.0, -1))
        nitrogenNN.angular_symmetry_functions.append((4.0, 3.5, -1))
        nitrogenNN.angular_symmetry_functions.append((4.0, 4.0, -1))
        nitrogenNN.angular_symmetry_functions.append((4.0, 4.5, 1))
        nitrogenNN.angular_symmetry_functions.append((4.0, 5.0, 1))
        nitrogenNN.angular_symmetry_functions.append((4.0, 5.5, 1))

    def initialize_FF3(self):
        # create a dictionary to look up force field object for each element
        self.element_force_field = {}

        # for now this is just a list of tuples that define the Gaussian width
        # shifts, and cutoff value for each Gaussian symmetry function.  Eventually, this will be
        # a more complicated object...

        gauss_width_min = 0.5
        gauss_width_max = 2.0
        num_widths = 10

        cutoff_min = 3.0
        cutoff_max = 6.0
        num_cutoffs = 5

        gauss_vec = np.linspace(gauss_width_min, gauss_width_max, num_widths)
        cutoff_vec = np.linspace(cutoff_min, cutoff_max, num_cutoffs)
        print(range(num_widths))
        # setup force field for oxygen
        oxygenNN = elementNN()
        self.element_force_field["O"] = oxygenNN
        # add list of symmetry function tuples
        for i in range(num_widths):
            for j in range(num_cutoffs):
                oxygenNN.symmetry_functions.append((gauss_vec[i],
                                                    cutoff_vec[j]))

        # setup force field for hydrogen
        hydrogenNN = elementNN()
        self.element_force_field["H"] = hydrogenNN
        # add list of symmetry function tuples
        for i in range(num_widths):
            for j in range(num_cutoffs):
                hydrogenNN.symmetry_functions.append((gauss_vec[i],
                                                      cutoff_vec[j]))

    def initialize_GA_opt(self):
        """Initializes symmetry function parameters.

        These specific parameters were chosen by the genetic algorithm
        optimized symmetry functions described in the wACSF paper. They
        tend to work very well and are conveniently consistent across
        atomtypes.

        Some compressed parameter --> NN method might be useful in the future.
        
        Unless we do our own genetic algorithm optimization on particular 
        systems (or just gradient descent as long as we can concede 
        constant parameter count), then we should just stick with these.
        Exhaustively looking for better parameters with grid search 
        is intractable.
        
        """

        # create a dictionary to look up force field object for each element
        self.element_force_field = {}

        hydrogenNN = elementNN()
        self.element_force_field["H"] = hydrogenNN
        # add list of symmetry function tuples
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 7.5))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 7.22))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 6.94))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 6.66))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 6.38))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 6.10))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 5.82))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 5.54))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 5.26))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 4.98))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 4.70))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 4.42))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 4.14))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 3.86))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 3.58))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 3.30))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 3.02))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 2.74))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 2.46))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 2.18))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 1.90))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 1.62))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 1.34))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 1.06))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 0.78))
        hydrogenNN.radial_symmetry_functions.append((6.3775510, 0.50))

        hydrogenNN.angular_symmetry_functions.append((0.0836777, 0.0, -1))
        hydrogenNN.angular_symmetry_functions.append((0.0836777, 0.0, 1))
        hydrogenNN.angular_symmetry_functions.append((0.1685744, 0.0, -1))
        hydrogenNN.angular_symmetry_functions.append((0.1685744, 0.0, 1))
        hydrogenNN.angular_symmetry_functions.append((0.5, 0.0, -1))
        hydrogenNN.angular_symmetry_functions.append((0.5, 0.0, 1))

        oxygenNN = elementNN()
        self.element_force_field["O"] = oxygenNN
        # add list of symmetry function tuples
        oxygenNN.radial_symmetry_functions.append((6.3775510, 7.50))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 7.22))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 6.94))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 6.66))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 6.38))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 6.10))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 5.82))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 5.54))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 5.26))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 4.98))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 4.70))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 4.42))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 4.14))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 3.86))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 3.58))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 3.30))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 3.02))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 2.74))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 2.46))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 2.18))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 1.90))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 1.62))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 1.34))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 1.06))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 0.78))
        oxygenNN.radial_symmetry_functions.append((6.3775510, 0.50))

        oxygenNN.angular_symmetry_functions.append((0.0836777, 0, -1))
        oxygenNN.angular_symmetry_functions.append((0.0836777, 0, 1))
        oxygenNN.angular_symmetry_functions.append((0.1685744, 0, -1))
        oxygenNN.angular_symmetry_functions.append((0.1685744, 0, 1))
        oxygenNN.angular_symmetry_functions.append((0.5, 0, -1))
        oxygenNN.angular_symmetry_functions.append((0.5, 0, 1))

        carbonNN = elementNN()
        self.element_force_field["C"] = carbonNN
        # add list of symmetry function tuples
        carbonNN.radial_symmetry_functions.append((6.3775510, 7.50))
        carbonNN.radial_symmetry_functions.append((6.3775510, 7.22))
        carbonNN.radial_symmetry_functions.append((6.3775510, 6.94))
        carbonNN.radial_symmetry_functions.append((6.3775510, 6.66))
        carbonNN.radial_symmetry_functions.append((6.3775510, 6.38))
        carbonNN.radial_symmetry_functions.append((6.3775510, 6.10))
        carbonNN.radial_symmetry_functions.append((6.3775510, 5.82))
        carbonNN.radial_symmetry_functions.append((6.3775510, 5.54))
        carbonNN.radial_symmetry_functions.append((6.3775510, 5.26))
        carbonNN.radial_symmetry_functions.append((6.3775510, 4.98))
        carbonNN.radial_symmetry_functions.append((6.3775510, 4.70))
        carbonNN.radial_symmetry_functions.append((6.3775510, 4.42))
        carbonNN.radial_symmetry_functions.append((6.3775510, 4.14))
        carbonNN.radial_symmetry_functions.append((6.3775510, 3.86))
        carbonNN.radial_symmetry_functions.append((6.3775510, 3.58))
        carbonNN.radial_symmetry_functions.append((6.3775510, 3.30))
        carbonNN.radial_symmetry_functions.append((6.3775510, 3.02))
        carbonNN.radial_symmetry_functions.append((6.3775510, 2.74))
        carbonNN.radial_symmetry_functions.append((6.3775510, 2.46))
        carbonNN.radial_symmetry_functions.append((6.3775510, 2.18))
        carbonNN.radial_symmetry_functions.append((6.3775510, 1.90))
        carbonNN.radial_symmetry_functions.append((6.3775510, 1.62))
        carbonNN.radial_symmetry_functions.append((6.3775510, 1.34))
        carbonNN.radial_symmetry_functions.append((6.3775510, 1.06))
        carbonNN.radial_symmetry_functions.append((6.3775510, 0.78))
        carbonNN.radial_symmetry_functions.append((6.3775510, 0.50))

        carbonNN.angular_symmetry_functions.append((0.0836777, 0, -1))
        carbonNN.angular_symmetry_functions.append((0.0836777, 0, 1))
        carbonNN.angular_symmetry_functions.append((0.1685744, 0, -1))
        carbonNN.angular_symmetry_functions.append((0.1685744, 0, 1))
        carbonNN.angular_symmetry_functions.append((0.5, 0, -1))
        carbonNN.angular_symmetry_functions.append((0.5, 0, 1))

        nitrogenNN = elementNN()
        self.element_force_field["N"] = nitrogenNN
        # add list of symmetry function tuples
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 7.50))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 7.22))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 6.94))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 6.66))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 6.38))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 6.10))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 5.82))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 5.54))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 5.26))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 4.98))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 4.70))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 4.42))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 4.14))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 3.86))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 3.58))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 3.30))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 3.02))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 2.74))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 2.46))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 2.18))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 1.90))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 1.62))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 1.34))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 1.06))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 0.78))
        nitrogenNN.radial_symmetry_functions.append((6.3775510, 0.50))

        nitrogenNN.angular_symmetry_functions.append((0.0836777, 0, -1))
        nitrogenNN.angular_symmetry_functions.append((0.0836777, 0, 1))
        nitrogenNN.angular_symmetry_functions.append((0.1685744, 0, -1))
        nitrogenNN.angular_symmetry_functions.append((0.1685744, 0, 1))
        nitrogenNN.angular_symmetry_functions.append((0.5, 0, -1))
        nitrogenNN.angular_symmetry_functions.append((0.5, 0, 1))

        fluorineNN = elementNN()
        self.element_force_field["F"] = fluorineNN
        # add list of symmetry function tuples
        fluorineNN.radial_symmetry_functions.append((6.3775510, 7.50))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 7.22))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 6.94))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 6.66))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 6.38))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 6.10))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 5.82))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 5.54))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 5.26))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 4.98))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 4.70))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 4.42))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 4.14))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 3.86))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 3.58))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 3.30))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 3.02))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 2.74))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 2.46))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 2.18))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 1.90))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 1.62))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 1.34))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 1.06))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 0.78))
        fluorineNN.radial_symmetry_functions.append((6.3775510, 0.50))

        fluorineNN.angular_symmetry_functions.append((0.0836777, 0, -1))
        fluorineNN.angular_symmetry_functions.append((0.0836777, 0, 1))
        fluorineNN.angular_symmetry_functions.append((0.1685744, 0, -1))
        fluorineNN.angular_symmetry_functions.append((0.1685744, 0, 1))
        fluorineNN.angular_symmetry_functions.append((0.5, 0, -1))
        fluorineNN.angular_symmetry_functions.append((0.5, 0, 1))

    def initialize_solo_GA_opt(self):
        """Initialize single-network symfun parameters.

        This lets all elements share a single neural network, in theory
        allowing cross talk for information they share. This style simply
        encodes the element type as a one-hot encoded vector as an addition
        to the input descriptor.

        In practice, this is a really bad idea and the network has a hard
        time discerning useful information at the quantity of data we operated
        at (~500-50,000 points). In the future, a more sophisticated scheme
        allowing for cross talk might be beneficial.

        """

        # create a dictionary to look up force field object for each element
        self.element_force_field = {}

        all_elemNN = elementNN()
        self.element_force_field["H"] = all_elemNN
        self.element_force_field["O"] = all_elemNN
        self.element_force_field["C"] = all_elemNN
        self.element_force_field["N"] = all_elemNN

        # add list of symmetry function tuples
        all_elemNN.radial_symmetry_functions.append((6.3775510, 7.5))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 7.22))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 6.94))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 6.66))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 6.38))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 6.10))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 5.82))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 5.54))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 5.26))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 4.98))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 4.70))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 4.42))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 4.14))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 3.86))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 3.58))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 3.30))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 3.02))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 2.74))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 2.46))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 2.18))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 1.90))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 1.62))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 1.34))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 1.06))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 0.78))
        all_elemNN.radial_symmetry_functions.append((6.3775510, 0.50))

        all_elemNN.angular_symmetry_functions.append((0.0836777, 0.0, -1))
        all_elemNN.angular_symmetry_functions.append((0.0836777, 0.0, 1))
        all_elemNN.angular_symmetry_functions.append((0.1685744, 0.0, -1))
        all_elemNN.angular_symmetry_functions.append((0.1685744, 0.0, 1))
        all_elemNN.angular_symmetry_functions.append((0.5, 0.0, -1))
        all_elemNN.angular_symmetry_functions.append((0.5, 0.0, 1))
