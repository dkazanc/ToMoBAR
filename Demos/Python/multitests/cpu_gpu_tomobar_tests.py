
"""A script to test cpu and gpu functionality by running main 2D/3D algorithms

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev
"""
import os
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.artifacts import _Artifacts_
from tomobar.methodsIR import RecToolsIR
from tomobar.methodsDIR import RecToolsDIR

print ("Building 3D phantom using TomoPhantom software")
model = 13 # select a model number from the library
N_size = 64 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2)*N_size) # detector column count (horizontal)
Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.25*np.pi*N_size) # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)

print ("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)

# adding noise
_noise_ = {'noise_type' : 'Poisson',
            'noise_sigma' : 8000, # noise amplitude
            'noise_seed' : 0}

projData3D_analyt_noise = _Artifacts_(projData3D_analyt, **_noise_)


print ("--------------------------------------------------------------")
print ("-----------------------CPU TESTS------------------------------")
print ("--------------------------------------------------------------")

RectoolsDirect = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector = 'cpu')
print ("--------------------------------------------------------------")
print ("Perform Fourier recon of 2D slice (cpu)")
Rec = RectoolsDirect.FOURIER(projData3D_analyt_noise[32,:,:],'linear')
print ("Done...")
print ("--------------------------------------------------------------")
print ("Perform FBP recon of 2D slice (cpu)")
Rec = RectoolsDirect.FBP(projData3D_analyt_noise[32,:,:])
print ("Done...")

RectoolsIterative = RecToolsIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity = 'LS',
                    device_projector = 'cpu')
print ("--------------------------------------------------------------")
print ("Perform iterative recon (FISTA) of 2D slice (cpu)")
# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : projData3D_analyt_noise[32,:,:]} # data dictionary
lc = RectoolsIterative.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {'iterations' : 2,
               'lipschitz_const' : lc}
# Run FISTA reconstrucion algorithm without regularisation
Rec = RectoolsIterative.FISTA(_data_, _algorithm_, {})
print ("Done...")
print ("--------------------------------------------------------------")
print ("Perform REGULARISED iterative recon (FISTA) of 2D slice (cpu)")
# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.0005,
                    'iterations' : 5,
                    'device_regulariser': 'cpu'}
Rec = RectoolsIterative.FISTA(_data_, _algorithm_, _regularisation_)
print ("Done...")

RectoolsDirect = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector = 'cpu')
print ("--------------------------------------------------------------")
print ("Perform FBP3D recon (cpu)")
Rec = RectoolsDirect.FBP(projData3D_analyt_noise)
print ("Done...")

print ("--------------------------------------------------------------")
print ("-----------------------GPU TESTS------------------------------")
print ("--------------------------------------------------------------")

RectoolsDirect = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector = 'gpu')
print ("--------------------------------------------------------------")
print ("Perform FBP 2D recon (gpu)")
Rec = RectoolsDirect.FBP(projData3D_analyt_noise[32,:,:])
print ("Done...")

RectoolsDirect = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector = 'gpu')
print ("--------------------------------------------------------------")
print ("Perform FBP3D recon (gpu)")
Rec = RectoolsDirect.FBP(projData3D_analyt_noise)
print ("Done...")

RectoolsIterative = RecToolsIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity = 'LS',
                    device_projector = 'gpu')


_data_ = {'projection_norm_data' : projData3D_analyt_noise[32,:,:]} # data dictionary
lc = RectoolsIterative.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {'iterations' : 2,
               'lipschitz_const' : lc}
# Run FISTA reconstrucion algorithm without regularisation
print ("--------------------------------------------------------------")
print ("Perform iterative recon (FISTA) of 2D slice (gpu)")
Rec = RectoolsIterative.FISTA(_data_, _algorithm_, {})
print ("Done...")

print ("--------------------------------------------------------------")
print ("Perform REGULARISED iterative recon (FISTA) of 2D slice (gpu)")
# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.0005,
                    'iterations' : 5,
                    'device_regulariser': 'gpu'}
Rec = RectoolsIterative.FISTA(_data_, _algorithm_, _regularisation_)
print ("Done...")

RectoolsIterative = RecToolsIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity = 'LS',
                    device_projector = 'gpu')


_data_ = {'projection_norm_data' : projData3D_analyt_noise} # data dictionary
lc = RectoolsIterative.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {'iterations' : 2,
               'lipschitz_const' : lc}
print ("--------------------------------------------------------------")
print ("Perform REGULARISED iterative recon (FISTA) of 3D data (gpu)")
# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.0005,
                    'iterations' : 5,
                    'device_regulariser': 'gpu'}
Rec = RectoolsIterative.FISTA(_data_, _algorithm_, _regularisation_)
print ("Done...")