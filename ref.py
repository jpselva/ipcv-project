import numpy as np


def create3dRef(face_points):
    nose = face_points['nose']
    cheek_l = face_points['cheek_l']
    cheek_r = face_points['cheek_r']

    origin = nose

    y = np.cross(cheek_r - nose, cheek_l - nose)
    x = cheek_l - cheek_r
    z = -np.cross(x, y)  # to follow right hand rule

    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)

    return origin, x, y, z


def convertToRef(point, origin, x, y, z):
    v = point - origin

    x_proj = np.dot(v, x)
    y_proj = np.dot(v, y)
    z_proj = np.dot(v, z)

    return x_proj, y_proj, z_proj
