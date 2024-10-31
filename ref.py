from triangulation import triangulate_points

def create3dRef(frame, reference_points_c1, reference_points_c2, R, T):
    #TODO: implement this function
    #Idea for now:
    # 1. get origin and x points from both cameras
    # 2. triangulate the points to get the 3D points
    # 3. calclate x vector in 3d space
    # 4. calculate y vector like (xy, -xx, xz) to get a perpendicular vector to x
    # 5. calculate z vector by cross product of x and y
    # 6. make a 3d plot of the vectors and interest point (in another function probably)
    
    #print(reference_points_c1, reference_points_c2) -> [(x1, y1), (x2, y2), ...] [(x1, y1), (x2, y2), ...]

    points3d = triangulate_points(reference_points_c1, reference_points_c2, R, T)
    origin3d = points3d[0]
    x3d = points3d[1]
    print("Origin:", origin3d)
    print("X:", x3d)
    return