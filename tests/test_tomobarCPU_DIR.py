import unittest
import numpy as np
from tomobar.methodsDIR import RecToolsDIR
from tomobar.methodsIR import RecToolsIR

###############################################################################
class TestTomobar(unittest.TestCase):

    def test_tomobarDIR(self):
        model = 4 # select a model
        N_size = 64 # set dimension of the phantom

        # create sinogram analytically
        angles_num = int(0.5*np.pi*N_size); # angles number
        angles = np.linspace(0.0,179.9,angles_num,dtype='float32')
        angles_rad = angles*(np.pi/180.0)
        P = int(np.sqrt(2)*N_size) # detectors
        sino_num = np.ones((P, angles_num))

        RectoolsDirect = RecToolsDIR(DetectorsDimH = P,  # DetectorsDimH # detector dimension (horizontal)
                            DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                            CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                            AnglesVec = angles_rad, # array of angles in radians
                            ObjSize = N_size, # a scalar to define reconstructed object dimensions
                            device_projector = 'cpu')
        """
        RectoolsIterative = RecToolsIR(DetectorsDimH = P,  # DetectorsDimH # detector dimension (horizontal)
                            DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                            CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                            AnglesVec = angles_rad, # array of angles in radians
                            ObjSize = N_size, # a scalar to define reconstructed object dimensions
                            datafidelity = 'LS',
                            device_projector = 'cpu')
        """
        RecFourier = RectoolsDirect.FOURIER(sino_num,'linear')

###############################################################################
if __name__ == '__main__':
    unittest.main()
