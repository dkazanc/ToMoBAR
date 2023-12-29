"""The base reconstruction class for the varierty of direct and iterative methods.

@authors: Daniil Kazantsev: https://github.com/dkazanc
"""
astra_enabled = False
try:
    from tomobar.supp.astraOP import AstraTools, AstraTools3D

    astra_enabled = True
except ImportError:
    print("____! Astra-toolbox package is missing, please install !____")

from tomobar.supp.astraOP import parse_device_argument
from tomobar.supp.suppTools import swap_data_axis_to_accepted


class RecTools:
    """The base reconstruction class
    
    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        device_projector (str, int): 'cpu' or 'gpu'  device OR provide a GPU index (integer) of a specific GPU device.
        data_axis_labels (list): a list with axis labels of the input data, e.g. ['detY', 'angles', 'detX'].
    """

    def __init__(
        self,
        DetectorsDimH,  # DetectorsDimH # detector dimension (horizontal)
        DetectorsDimV,  # DetectorsDimV # detector dimension (vertical) for 3D case only
        CenterRotOffset,  # Centre of Rotation (CoR) scalar or a vector
        AnglesVec,  # Array of angles in radians
        ObjSize,  # A scalar to define reconstructed object dimensions
        device_projector="gpu",  # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
        data_axis_labels=None,  # the input data axis labels
    ):
        if isinstance(ObjSize, tuple):
            raise ValueError(
                " Reconstruction is currently available for square or cubic objects only, please provide a scalar"
            )
        else:
            self.ObjSize = ObjSize  # size of the object
        self.DetectorsDimH = DetectorsDimH
        if DetectorsDimV == 0:
            self.DetectorsDimV = None
        else:
            self.DetectorsDimV = DetectorsDimV
        self.AnglesVec = AnglesVec
        self.angles_number = len(AnglesVec)
        if CenterRotOffset is not None:
            self.CenterRotOffset = CenterRotOffset
        else:
            self.CenterRotOffset = 0.0
        self.datafidelity = "None"
        self.device_projector, self.GPUdevice_index = parse_device_argument(
            device_projector
        )
        self.data_axis_labels = data_axis_labels
        if self.DetectorsDimV is None:
            # 2D geometry
            # silently re-initialise the default for 2D case
            if data_axis_labels is None:
                self.data_axis_labels = ["angles", "detX"]
            if len(self.data_axis_labels) == 3:
                print(
                    f"Warning: The labels {data_axis_labels} were provided for 3D geometry while the input data is 2D!"
                )
            self.data_swap_list = swap_data_axis_to_accepted(
                self.data_axis_labels, labels_order=["angles", "detX"]
            )
            self.geom = "2D"
            if astra_enabled:
                # initiate 2D ASTRA class object
                self.Atools = AstraTools(
                    self.DetectorsDimH,
                    self.AnglesVec,
                    self.CenterRotOffset,
                    self.ObjSize,
                    1,  # assuming no subsets
                    self.device_projector,
                    self.GPUdevice_index,
                )
        else:
            if data_axis_labels is None:
                self.data_axis_labels = [
                    "detY",
                    "angles",
                    "detX",
                ]  # silently re-initialise for 3D case
            if len(self.data_axis_labels) == 2:
                print(
                    f"Warning! The labels {data_axis_labels} were provided for 2D geometry while the input data is 3D!"
                )
            if "DIRCuPy" in self.__class__.__name__:
                # NOTE: the order of the accepted axis is different here
                # compared to the FBP implementation in methodsDIR. This is
                # due to filter has been re-implemented in more optimal fashion.
                # It shouldn't affect the user.
                required_order = ["angles", "detY", "detX"]
            else:
                required_order = ["detY", "angles", "detX"]
            self.data_swap_list = swap_data_axis_to_accepted(
                self.data_axis_labels, labels_order=required_order
            )
            self.geom = "3D"
            if astra_enabled:
                # initiate 3D ASTRA class object
                self.Atools = AstraTools3D(
                    self.DetectorsDimH,
                    self.DetectorsDimV,
                    self.AnglesVec,
                    self.CenterRotOffset,
                    self.ObjSize,
                    1,  # assuming no subsets
                    self.device_projector,
                    self.GPUdevice_index,
                )
