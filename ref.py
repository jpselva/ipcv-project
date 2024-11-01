from triangulation import triangulate_points
import numpy as np
import matplotlib.pyplot as plt
from draw import draw3dRef

def create3dRef(frame, reference_points_c1, reference_points_c2, R, T) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points3d_homog = triangulate_points(reference_points_c1, reference_points_c2, R, T)
    origin3d = points3d_homog[0][0:3]
    x3d = points3d_homog[1][0:3]
    
    x_vector = (x3d - origin3d)
    x_vector = x_vector / np.linalg.norm(x_vector)
    
    y_vector = np.array([x_vector[1], -x_vector[0], x_vector[2]], dtype=np.float32)
    y_vector = y_vector / np.linalg.norm(y_vector)
    
    z_vector = np.cross(x_vector, y_vector)
    z_vector = z_vector / np.linalg.norm(z_vector)

    return origin3d, x_vector, y_vector, z_vector