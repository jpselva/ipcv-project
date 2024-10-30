from triangulation import triangulate_points

def create3dRef(frame, reference_points):
    #TODO: implement this function
    #Idea for now:
    # 1. get origin and x points from both cameras
    # 2. triangulate the points to get the 3D points
    # 3. calclate x vector in 3d space
    # 4. calculate y vector like (xy, -xx, xz) to get a perpendicular vector to x
    # 5. calculate z vector by cross product of x and y
    # 6. Project x and y vector to the image plane (use drawVector)
    pass
