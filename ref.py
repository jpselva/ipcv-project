import numpy as np


def create3dRef(face_points):
    nose = face_points['nose']
    cheek_l = face_points['cheek_l']
    cheek_r = face_points['cheek_r']

    origin = nose

    y = np.cross(cheek_r - nose, cheek_l - nose)
    x = cheek_l - cheek_r
    z = -np.cross(x, y)

    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)

    return origin, x, y, z
