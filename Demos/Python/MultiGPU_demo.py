"""A script to demonstrate multi-gpu capabilities of ToMoBAR package, note that mpi4py is required.
Used reference: https://github.com/mcduta/programming/tree/main/simple-mpi-cuda

# Run the script as:
# mpirun -np 2 python MultiGPU_demo.py -g -s -gpus 1
# where:
#   -np is a number of processes
#   -gpus is the number of GPUs available


GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev
"""


def data_generator():
    import os
    import numpy as np
    import tomophantom
    from tomophantom import TomoP3D
    from tomophantom.supp.artifacts import _Artifacts_

    print("Building 3D phantom using TomoPhantom software")
    data_dict = {}
    model = 13  # select a model number from the library
    N_size = 128  # Define phantom dimensions using a scalar value (cubic phantom)
    path = os.path.dirname(tomophantom.__file__)
    path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

    # Projection geometry related parameters:
    Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
    Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
    angles_num = int(0.25 * np.pi * N_size)  # angles number
    angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
    angles_rad = angles * (np.pi / 180.0)

    print("Generate 3D analytical projection data with TomoPhantom")
    projData3D_analyt = TomoP3D.ModelSino(
        model, N_size, Horiz_det, Vert_det, angles, path_library3D
    )

    # adding noise
    _noise_ = {
        "noise_type": "Poisson",
        "noise_sigma": 8000,  # noise amplitude
        "noise_seed": 0,
    }

    projData3D_analyt_noise = _Artifacts_(projData3D_analyt, **_noise_)

    del projData3D_analyt
    data_dict["model"] = model
    data_dict["N_size"] = N_size
    data_dict["Horiz_det"] = Horiz_det
    data_dict["Vert_det"] = Vert_det
    data_dict["angles_num"] = angles_num
    data_dict["angles_rad"] = angles_rad
    data_dict["proj_data"] = projData3D_analyt_noise
    return data_dict


def reconstructorSIRT(data_proj, iterations_alg, DEVICE_no):
    # perform basic data splitting between GPUs
    print("-----------------------------------------------------------------")
    print(
        "Perform SIRT reconstruction in parallel on {} GPU device...".format(DEVICE_no)
    )
    print("-----------------------------------------------------------------")
    from tomobar.methodsIR import RecToolsIR

    # set parameters and initiate a class object
    Rectools = RecToolsIR(
        DetectorsDimH=data_dict["Horiz_det"],  # Horizontal detector dimension
        DetectorsDimV=data_dict["Vert_det"],  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=data_dict["angles_rad"],  # A vector of projection angles in radians
        ObjSize=data_dict["N_size"],  # Reconstructed object dimensions (scalar)
        datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
        device_projector=DEVICE_no,
    )

    # prepare dictionaries with parameters:
    _data_ = {"projection_norm_data": data_dict["proj_data"]}  # data dictionary
    _algorithm_ = {"iterations": iterations_alg}

    RecSIRT = Rectools.SIRT(_data_, _algorithm_)  # SIRT reconstruction
    return RecSIRT


def reconstructorFISTA(data_proj, iterations_alg, DEVICE_no):
    # perform basic data splitting between GPUs
    print("-----------------------------------------------------------------")
    print(
        "Perform FISTA reconstruction in parallel on {} GPU device...".format(DEVICE_no)
    )
    print("-----------------------------------------------------------------")
    from tomobar.methodsIR import RecToolsIR

    # set parameters and initiate a class object
    Rectools = RecToolsIR(
        DetectorsDimH=data_dict["Horiz_det"],  # Horizontal detector dimension
        DetectorsDimV=data_dict["Vert_det"],  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=data_dict["angles_rad"],  # A vector of projection angles in radians
        ObjSize=data_dict["N_size"],  # Reconstructed object dimensions (scalar)
        datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
        device_projector=DEVICE_no,
    )

    # prepare dictionaries with parameters:
    _data_ = {
        "projection_norm_data": data_dict["proj_data"],
        "OS_number": 8,
    }  # data dictionary

    lc = Rectools.powermethod(
        _data_
    )  # calculate Lipschitz constant (run once to initialise)

    # Run FISTA reconstrucion algorithm without regularisation
    _algorithm_ = {"iterations": iterations_alg, "lipschitz_const": lc}

    # adding regularisation using the CCPi regularisation toolkit
    _regularisation_ = {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 200,
        "device_regulariser": DEVICE_no,
    }

    # Run FISTA reconstrucion algorithm with 3D regularisation
    RecFISTA_os_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
    return RecFISTA_os_reg


# %%
if __name__ == "__main__":
    # imports
    from mpi4py import MPI

    # MPI process
    mpi_proc_num = MPI.COMM_WORLD.size
    mpi_proc_id = MPI.COMM_WORLD.rank

    # process arguments
    import argparse

    parser = argparse.ArgumentParser(description="GPU device use from mpi4py")
    parser.add_argument(
        "-g",
        "--get_device",
        action="store_true",
        help="report device for each MPI process (default: NO)",
    )
    parser.add_argument(
        "-s",
        "--set_device",
        action="store_true",
        help="automatically set device for each MPI process (default: NO)",
    )
    parser.add_argument(
        "-gpus",
        "--gpus_no",
        dest="gpus_total",
        default=2,
        help="the total number of available GPU devices",
    )
    args = parser.parse_args()

    # Generating the projection data
    # NOTE that the data is generated for each process for the sake of simplicity but it could be splitted into multiple mpi processess generating smaller chunks of the global dataset
    data_dict = data_generator()

    # set the total number of available GPU devices
    GPUs_total_num = int(args.gpus_total)
    data_dict["gpu_devices_total"] = GPUs_total_num
    DEVICE_no = mpi_proc_id % GPUs_total_num

    # reconstructing using SIRT algorithm:
    iterations_alg = 500
    reconstructionSIRT = reconstructorSIRT(data_dict, iterations_alg, DEVICE_no)

    # reconstructing using the regularised FISTA algorithm:
    iterations_alg = 5
    reconstructionFISTA = reconstructorFISTA(data_dict, iterations_alg, DEVICE_no)
# %%
