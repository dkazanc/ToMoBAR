"""Supporting functions

"""
import numpy as np
from typing import Union, List

# define a 2D vector geometry
def _vec_geom_init2D(angles_rad: np.ndarray, 
                      CenterRotOffset: Union[float, List]) -> np.ndarray:
    DetectorSpacingX = 1.0
    s0 = [0.0, -1.0]  # source
    u0 = [DetectorSpacingX, 0.0]  # detector coordinates
    vectors = np.zeros([angles_rad.size, 6])
    for i in range(0, angles_rad.size):
        if np.ndim(CenterRotOffset) == 0:
            d0 = [CenterRotOffset, 0.0]  # detector
        else:
            d0 = [CenterRotOffset[i], 0.0]  # detector
        theta = angles_rad[i]
        vec_temp = np.dot(__rotation_matrix2D(theta), s0)
        vectors[i, 0:2] = vec_temp[:]  # ray position
        vec_temp = np.dot(__rotation_matrix2D(theta), d0)
        vectors[i, 2:4] = vec_temp[:]  # center of detector position
        vec_temp = np.dot(__rotation_matrix2D(theta), u0)
        vectors[i, 4:6] = vec_temp[:]  # detector pixel (0,0) to (0,1).
    return vectors

# define 3D vector geometry
def _vec_geom_init3D(angles_rad, DetectorSpacingX, DetectorSpacingY, CenterRotOffset):
    s0 = [0.0, -1.0, 0.0]  # source
    u0 = [DetectorSpacingX, 0.0, 0.0]  # detector coordinates
    v0 = [0.0, 0.0, DetectorSpacingY]  # detector coordinates

    vectors = np.zeros([angles_rad.size, 12])
    for i in range(0, angles_rad.size):
        if np.ndim(CenterRotOffset) == 0:
            d0 = [CenterRotOffset, 0.0, 0.0]  # detector
        else:
            d0 = [CenterRotOffset[i, 0], 0.0, CenterRotOffset[i, 1]]  # detector
        theta = angles_rad[i]
        vec_temp = np.dot(__rotation_matrix3D(theta), s0)
        vectors[i, 0:3] = vec_temp[:]  # ray position
        vec_temp = np.dot(__rotation_matrix3D(theta), d0)
        vectors[i, 3:6] = vec_temp[:]  # center of detector position
        vec_temp = np.dot(__rotation_matrix3D(theta), u0)
        vectors[i, 6:9] = vec_temp[:]  # detector pixel (0,0) to (0,1).
        vec_temp = np.dot(__rotation_matrix3D(theta), v0)
        vectors[i, 9:12] = vec_temp[:]  # Vector from detector pixel (0,0) to (1,0)
    return vectors


# define 2D rotation matrix
def __rotation_matrix2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# define 3D rotation matrix
def __rotation_matrix3D(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
