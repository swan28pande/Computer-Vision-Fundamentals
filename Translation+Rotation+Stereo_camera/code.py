
"""
3D Computer Vision: 3D Object Rendering with Translation and Rotation

This program demonstrates advanced computer vision techniques for rendering 3D objects
with realistic motion, texture mapping, and stereo vision capabilities.

Features:
- 3D cube rendering with rotation and translation
- Texture mapping using image files
- Stereo vision simulation with dual camera setup
- Real-time animation with physics-based motion

Requirements:
- Python 3.8+
- OpenCV (cv2): pip install opencv-python
- NumPy: pip install numpy

Author: Computer Vision Fundamentals Project
"""

import sys
import numpy as np
import cv2

# Core computer vision functions for 3D to 2D projection and rendering

"""
3D to 2D Projection Function

Projects 3D world coordinates to 2D image coordinates using camera intrinsic
parameters, rotation matrix, and translation vector.

Args:
    K: 3x3 intrinsic camera matrix
    R: 3x3 rotation matrix
    T: 3x1 translation vector
    Vi: 3D point coordinates (X, Y, Z)

Returns:
    v: 2D image coordinates (x, y) in normalized camera coordinates
"""
def Map2Da(K, R, T, Vi):
    T_transpose = np.transpose(np.atleast_2d(T)) #numpy needs to treat 1D as 2D to transpose
    V_transpose = np.transpose(np.atleast_2d(np.append(Vi,[1])))
    RandTappended = np.append(R, T_transpose, axis=1)
    P = K @ RandTappended @ V_transpose #@ is the matrix mult operator for numpy arrays
    P = np.asarray(P).flatten() #just to make it into a flat array

    w1 = P[2]
    v= [None]*2 #makes an empty array of size 2

    #map Vi = (X, Y, Z) to v = (x, y)
    v[0]= P[0] / w1  #v[0] is the x-value for the 2D point v

    #MISSING: compute v[1], the y-value for the 2D point v
    v[1] = P[1]/w1

    return v


"""
Image Coordinate Mapping Function

Converts normalized camera coordinates (in mm) to pixel coordinates
for image rendering. Handles the coordinate system transformation
from camera space to image pixel space.

Args:
    u: 2D point in camera coordinate system (mm)
    c0: Image center column coordinate (pixels)
    r0: Image center row coordinate (pixels)
    p: Pixel size in mm

Returns:
    v: 2D point in pixel coordinates [row, col]
"""
def MapIndex(u, c0, r0, p):
    v = [None]*2
    v[0] = round(r0 - u[1]/p)
    # Note: In image coordinate system --
    #       a) the first index represents "row", the vertical position (y), the positive direction is downward, 
    #          i.e. for the pixel higher than the center, its first index is smaller than r0;
    #       b) the second index represents "column", the horizontal position (x), the positive direction is to the right
    #          i.e. for the pixel on the right side of the center, its second index is larger than c0;
    # MISSING: complete the line below:
    v[1] = round(c0+u[0]/p)
    return v

"""
Custom Line Drawing Function

Implements Bresenham's line algorithm for drawing lines directly on image arrays.
This custom implementation provides better control over line rendering compared
to OpenCV's built-in line function.

Args:
    A: Image array to draw the line on
    vertex1: Starting point of the line [row, col]
    vertex2: Ending point of the line [row, col]
    color: BGR color tuple (default: red)
    thickness: Line thickness in pixels (default: 3)

Returns:
    A: Modified image array with the line drawn

Note:
    The function handles coordinate system differences between camera coordinates
    and image pixel coordinates automatically.
"""

#MISSING : Replace the function below with another one that does not call
# cv2.line(.) but does all calculations within itself.
def drawLine(A,vertex1, vertex2, color = (0, 0, 255), thickness=3):
    # v1 = list(reversed(vertex1))
    # v2 = list(reversed(vertex2))
    # Note: After Map2Da() and MapIndex(), the input vertex1 and vertex2 are positions on the image,
    #       i.e. their first indices represent vertical positions (y), and second are horizontal ones (x). 
    #       However, the built-in function cv2.line() needs the inputs v1 and v2 to have the form (x, y), 
    #       that's the reason for which the above two lines use reversed() function.
    #       When you implement your own drawing functions, you have two options:
    #       a) you can keep the above two lines and exchange the indices back to (y, x) when drawing the line, 
    #       e.g. A[v[1], v[0]] = color;
    #       b) comment out these two lines, and still use A[v[0], v[1]] = color; 
    # Extract coordinates: vertex = [row, col] = [y, x]
    y1, x1 = vertex1[0], vertex1[1]
    y2, x2 = vertex2[0], vertex2[1]
    
    if y1 == y2:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 <= y1 < A.shape[0] and 0 <= x < A.shape[1]:
                A[y1, x] = color
        return A
    
    if x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 <= y < A.shape[0] and 0 <= x1 < A.shape[1]:
                A[y, x1] = color
        return A
    
    if abs(x2 - x1) >= abs(y2 - y1):
        if x1 > x2:  
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        m = (y2 - y1) / (x2 - x1)  
        
        for x in range(x1, x2 + 1):
            y = int(y1 + m * (x - x1))  
            if 0 <= y < A.shape[0] and 0 <= x < A.shape[1]:
                A[y, x] = color
    else:
        if y1 > y2:  
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        m = (x2 - x1) / (y2 - y1) 
        
        for y in range(y1, y2 + 1):
            x = int(x1 + m * (y - y1))
            if 0 <= y < A.shape[0] and 0 <= x < A.shape[1]:
                A[y, x] = color

    return A
  



def main():
    """
    Main function demonstrating 3D computer vision techniques.
    
    This function implements three main tasks:
    1. Basic 3D cube rendering with rotation and translation
    2. Cube rendering with background and texture mapping
    3. Stereo vision simulation with dual camera setup
    """
    
    # =============================================================================
    # TASK 1: Basic 3D Cube Rendering
    # =============================================================================
    print("Starting Task 1: Basic 3D Cube Rendering")
    
    # Define cube geometry
    length = 10  # Edge length in mm
    #the 8 3D points of the cube in mm:
    V1 = np.array([0, 0, 0])
    V2 = np.array([0, length, 0])
    V3 = np.array([length, length, 0])
    V4 = np.array([length, 0, 0])
    V5 = np.array([length, 0, length])
    V6 = np.array([0, length, length])
    V7 = np.array([0, 0, length])
    V8 = np.array([length, length, length])

    # Compute rotation axis and rotation matrix components
    # The rotation axis is along the diagonal of the cube (V8-V1)
    # This creates a natural spinning motion around the cube's diagonal
    u81 = (V8-V1)/np.linalg.norm(V8-V1)

    n_x = u81[0]
    n_y = u81[1]
    n_z = u81[2]
    N = np.zeros((3, 3))

    N[0][1] = -n_x
    N[0][2] = n_y
    N[1][0] = n_z
    N[1][2] = -n_x
    N[2][0] = -n_y
    N[2][1] = n_x

    # Camera and motion parameters
    # These values define the camera setup and object motion characteristics:
    T0 = np.array([-20, -25, 500])  # origin of object coordinate system in mm
    f = 40  # focal length in mm
    velocity = np.array([2, 9, 7])  # translational velocity
    acc = np.array([0.0, -0.80, 0])  # acceleration
    theta0 = 0 #initial angle of rotation is 0 (in degrees)
    w0 = 20  # angular velocity in deg/sec
    p = 0.01  # pixel size(mm)
    Rows = 600  # image size
    Cols = 600  # image size
    r0 = np.round(Rows / 2) #x-value of center of image
    c0 = np.round(Cols / 2) #y-value of center of image
    time_range = np.arange(0.0, 24.2, 0.2)

    # Initialize camera intrinsic matrix
    # This matrix defines the camera's internal parameters (focal length, principal point)
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
   
    # =============================================================================
    # TEXTURE MAPPING SETUP
    # =============================================================================
    # This section sets up texture mapping for one face of the cube.
    # We map a 2D texture image onto the 3D face defined by vertices V1, V2, V3, V4.
    # Each pixel in the texture corresponds to a 3D point on the cube face.
    
    # Calculate face dimensions and orientation vectors
    # These vectors define how the texture is oriented on the cube face

    h = np.linalg.norm(V2-V1)
    w = np.linalg.norm(V4-V1)
    u21 = (V2-V1)/np.linalg.norm(V2-V1)
    u41 = (V4-V1)/np.linalg.norm(V4-V1)

    # Generate 3D coordinates for each texture pixel
    # This creates a mapping between texture coordinates and 3D world coordinates:
    tmap = cv2.imread('Images/einstein50x50v.jpg')  # texture map image
    if tmap is None:
        print("image file can not be found on path given. Exiting now")
        sys.exit(1)

    r, c, colors = tmap.shape
    # We keep three arrays of size (r, c) to store the (X, Y, Z) points cooresponding
    # to each pixel on the texture 
    X = np.zeros((r, c), dtype=np.float64)
    Y = np.zeros((r, c), dtype=np.float64)
    Z = np.zeros((r, c), dtype=np.float64)
    for i in range(0, r):
        for j in range(0, c):
            p1 = V1 + (i) * u21 * (h / r) + (j) * u41 * (w / c)
            X[i, j] = p1[0]
            # Extract Y and Z coordinates for the 3D point
            Y[i, j] = p1[1]
            Z[i, j] = p1[2]

    # =============================================================================
    # TASK 1: Basic 3D Cube Rendering
    # =============================================================================
    print("Rendering Task 1: Basic 3D cube with rotation and translation...")
    
    for t in time_range:  # Generate animation frames over time
        theta = theta0 + w0 * t
        T = T0 + velocity * t + 0.5 * acc * t * t
        # Compute rotation matrix using Rodrigues' rotation formula
        # This creates smooth rotation around the specified axis
        theta_radians = theta * np.pi / 180  # Convert to radians
        R = np.identity(3) + np.sin(theta_radians) * N + (1 - np.cos(theta_radians)) * (N @ N)

        # find the image position of vertices

        # Project 3D cube vertices to 2D image coordinates
        # Each vertex is transformed through the camera pipeline:
        # 3D World -> Camera -> Image -> Pixel coordinates
        v = Map2Da(K, R, T, V1)
        v1 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V2)
        v2 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V3)
        v3 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V4)
        v4 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V5)
        v5 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V6)
        v6 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V7)
        v7 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V8)
        v8 = MapIndex(v, c0, r0, p)
        

        # Draw edges of the cube

        #color = (0, 0, 255) #note, CV uses BGR by default, not RGB. This is Red.
        color = (0, 0, 255) #note, CV uses BGR by default, not gray=(R+G+B)/3. This is Red.
        thickness = 2
        A = np.zeros((Rows, Cols, 3), dtype=np.uint8) #array which stores the image at this time step; (Rows x Cols) pixels, 3 channels per pixel
        
        # Draw all 12 edges of the cube
        # Each edge connects two vertices of the cube:
        A = drawLine(A, v1, v2, color, thickness)

        A = drawLine(A, v1, v4, color, thickness)

        A = drawLine(A, v1, v7, color, thickness)

        A = drawLine(A, v2, v3, color, thickness)

        A = drawLine(A, v2, v6, color, thickness)

        A = drawLine(A, v3, v4, color, thickness)

        A = drawLine(A, v3, v8, color, thickness)

        A = drawLine(A, v4, v5, color, thickness)

        A = drawLine(A, v5, v7, color, thickness)

        A = drawLine(A, v5, v8, color, thickness)

        A = drawLine(A, v6, v7, color, thickness)

        A = drawLine(A, v6, v8, color, thickness)


        # Apply texture mapping to the front face of the cube
        # This maps the texture image onto the 3D face:
        for i in range(r):
            for j in range(c):
                p1 = [X[i, j], Y[i, j], Z[i, j]]

                #p1 now stores the world point on the cubic face which
                #corresponds to (i, j) on the texture

                # Project the 3D texture point to 2D image coordinates
                # This determines where this texture pixel should appear on screen
                (ir, jr) = MapIndex(Map2Da(K, R, T, p1), c0, r0, p)

                # Apply texture color if the point is within image bounds
                if ((ir >= 0) and (jr >= 0) and (ir < Rows) and (jr < Cols)):
                    tmapval = tmap[i, j, 2]  # Extract red channel from texture
                    A[ir, jr] = [0, 0, tmapval]  # Apply as red color


        # Display the rendered frame
        cv2.imshow("3D Computer Vision Demo", A)
        
        # Control animation speed
        # Uncomment cv2.waitKey(0) for frame-by-frame viewing
        # Default waits 1ms for smooth animation
        cv2.waitKey(1)




    # =============================================================================
    # TASK 2: Cube Rendering with Background
    # =============================================================================
    print("Rendering Task 2: Cube with background and texture mapping...")
    
    # Load background image
    background = cv2.imread('Images/background.jpg')

    # Resize background to match image dimensions
    background = cv2.resize(background, (Cols, Rows))
    for t in time_range:  # Generate a sequence of images as a function of time
        theta = theta0 + w0 * t
        T = T0 + velocity * t + 0.5 * acc * t * t
        # Compute rotation matrix using Rodrigues' rotation formula
        # This creates smooth rotation around the specified axis
        theta_radians = theta * np.pi / 180  # Convert to radians
        R = np.identity(3) + np.sin(theta_radians) * N + (1 - np.cos(theta_radians)) * (N @ N)

        # find the image position of vertices

        # Project 3D cube vertices to 2D image coordinates
        # Each vertex is transformed through the camera pipeline:
        # 3D World -> Camera -> Image -> Pixel coordinates
        v = Map2Da(K, R, T, V1)
        v1 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V2)
        v2 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V3)
        v3 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V4)
        v4 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V5)
        v5 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V6)
        v6 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V7)
        v7 = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T, V8)
        v8 = MapIndex(v, c0, r0, p)
        

        # Draw edges of the cube

        #color = (0, 0, 255) #note, CV uses BGR by default, not RGB. This is Red.
        color = (0, 0, 255) #note, CV uses BGR by default, not gray=(R+G+B)/3. This is Red.
        thickness = 2
        A = np.copy(background) 
        
        # Draw all 12 edges of the cube
        # Each edge connects two vertices of the cube:
        A = drawLine(A, v1, v2, color, thickness)

        A = drawLine(A, v1, v4, color, thickness)

        A = drawLine(A, v1, v7, color, thickness)

        A = drawLine(A, v2, v3, color, thickness)

        A = drawLine(A, v2, v6, color, thickness)

        A = drawLine(A, v3, v4, color, thickness)

        A = drawLine(A, v3, v8, color, thickness)

        A = drawLine(A, v4, v5, color, thickness)

        A = drawLine(A, v5, v7, color, thickness)

        A = drawLine(A, v5, v8, color, thickness)

        A = drawLine(A, v6, v7, color, thickness)

        A = drawLine(A, v6, v8, color, thickness)


        # Apply texture mapping to the front face of the cube
        # This maps the texture image onto the 3D face:
        for i in range(r):
            for j in range(c):
                p1 = [X[i, j], Y[i, j], Z[i, j]]

                #p1 now stores the world point on the cubic face which
                #corresponds to (i, j) on the texture

                # Project the 3D texture point to 2D image coordinates
                # This determines where this texture pixel should appear on screen
                (ir, jr) = MapIndex(Map2Da(K, R, T, p1), c0, r0, p)

                # Apply texture color if the point is within image bounds
                if ((ir >= 0) and (jr >= 0) and (ir < Rows) and (jr < Cols)):
                    tmapval = tmap[i, j, 2]  # Extract red channel from texture
                    A[ir, jr] = [0, 0, tmapval]  # Apply as red color


        # Display the rendered frame
        cv2.imshow("3D Computer Vision Demo", A)
        
        # Control animation speed
        # Uncomment cv2.waitKey(0) for frame-by-frame viewing
        # Default waits 1ms for smooth animation
        cv2.waitKey(1)



    # =============================================================================
    # TASK 3: Stereo Vision Simulation
    # =============================================================================
    print("Rendering Task 3: Stereo vision with dual camera setup...")
    
    # Load background image
    background = cv2.imread('Images/background.jpg')

    # Resize background to match image dimensions
    background = cv2.resize(background, (Cols, Rows))
    
    for t in time_range:  # Generate a sequence of images as a function of time
        theta = theta0 + w0 * t
        T = T0 + velocity * t + 0.5 * acc * t * t
        # MISSING: compute rotation matrix R as shown in Eq. 2.34
        # Warning: be mindful of radians vs degrees
        # Note: for numpy data, @ operator can be used for dot product
        # Note: consistency of units for angle theta to be radians, you need convert it before computing
        #       sin(theta) and cos(theta).

        theta_radians = theta * np.pi / 180
        R = np.identity(3)+np.sin(theta_radians)*N+(1-np.cos(theta_radians))*(N@N)

        # find the image position of vertices

        #MISSING: given 3D vertices V1 to V8, map to 2D using Map2da
        #then, map to pixel space using mapindex
        #save all 2D vertices as v1 to v8

        # Stereo camera setup
        # Create two cameras separated by a baseline distance
        baseline = 10  # 10mm baseline between cameras
        T_left = T  # Left camera position
        T_right = T + np.array([baseline, 0, 0])  # Right camera position (shifted right)
        
        # Left camera vertices
        v = Map2Da(K, R, T_left, V1)
        v1_left = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_left, V2)
        v2_left = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_left, V3)
        v3_left = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_left, V4)
        v4_left = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_left, V5)
        v5_left = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_left, V6)
        v6_left = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_left, V7)
        v7_left = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_left, V8)
        v8_left = MapIndex(v, c0, r0, p)
        
        # Right camera vertices
        v = Map2Da(K, R, T_right, V1)
        v1_right = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_right, V2)
        v2_right = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_right, V3)
        v3_right = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_right, V4)
        v4_right = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_right, V5)
        v5_right = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_right, V6)
        v6_right = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_right, V7)
        v7_right = MapIndex(v, c0, r0, p)

        v = Map2Da(K, R, T_right, V8)
        v8_right = MapIndex(v, c0, r0, p)
        
        # Different colors for left and right camera views
        color_left = (0, 0, 255)   # Red for left camera (BGR format)
        color_right = (255, 0, 0)  # Blue for right camera (BGR format)
        thickness = 2
        
        A_left = np.copy(background)
        
        A_right = np.copy(background)
        
        # Draw edges for left camera (Red)
        A_left = drawLine(A_left, v1_left, v2_left, color_left, thickness)
        A_left = drawLine(A_left, v1_left, v4_left, color_left, thickness)
        A_left = drawLine(A_left, v1_left, v7_left, color_left, thickness)
        A_left = drawLine(A_left, v2_left, v3_left, color_left, thickness)
        A_left = drawLine(A_left, v2_left, v6_left, color_left, thickness)
        A_left = drawLine(A_left, v3_left, v4_left, color_left, thickness)
        A_left = drawLine(A_left, v3_left, v8_left, color_left, thickness)
        A_left = drawLine(A_left, v4_left, v5_left, color_left, thickness)
        A_left = drawLine(A_left, v5_left, v7_left, color_left, thickness)
        A_left = drawLine(A_left, v5_left, v8_left, color_left, thickness)
        A_left = drawLine(A_left, v6_left, v7_left, color_left, thickness)

        # Draw edges for right camera (Blue)
        A_right = drawLine(A_right, v1_right, v2_right, color_right, thickness)
        A_right = drawLine(A_right, v1_right, v4_right, color_right, thickness)
        A_right = drawLine(A_right, v1_right, v7_right, color_right, thickness)
        A_right = drawLine(A_right, v2_right, v3_right, color_right, thickness)
        A_right = drawLine(A_right, v2_right, v6_right, color_right, thickness)
        A_right = drawLine(A_right, v3_right, v4_right, color_right, thickness)
        A_right = drawLine(A_right, v3_right, v8_right, color_right, thickness)
        A_right = drawLine(A_right, v4_right, v5_right, color_right, thickness)
        A_right = drawLine(A_right, v5_right, v7_right, color_right, thickness)
        A_right = drawLine(A_right, v5_right, v8_right, color_right, thickness)
        A_right = drawLine(A_right, v6_right, v7_right, color_right, thickness)
        A_right = drawLine(A_right, v6_right, v8_right, color_right, thickness)
        


        for i in range(r):
            for j in range(c):
                p1 = [X[i, j], Y[i, j], Z[i, j]]

                #p1 now stores the world point on the cubic face which
                #corresponds to (i, j) on the texture

                # Map texture to left camera
                (ir_left, jr_left) = MapIndex(Map2Da(K, R, T_left, p1), c0, r0, p)
                ir_left, jr_left = int(ir_left), int(jr_left)

                if ((ir_left >= 0) and (jr_left >= 0) and (ir_left < Rows) and (jr_left < Cols)):
                    tmapval = tmap[i, j, 2]
                    A_left[ir_left, jr_left] = [0, 0, tmapval] # gray texture

                # Map texture to right camera
                (ir_right, jr_right) = MapIndex(Map2Da(K, R, T_right, p1), c0, r0, p)
                ir_right, jr_right = int(ir_right), int(jr_right)

                if ((ir_right >= 0) and (jr_right >= 0) and (ir_right < Rows) and (jr_right < Cols)):
                    tmapval = tmap[i, j, 2]
                    A_right[ir_right, jr_right] = [tmapval, 0, 0] # gray texture
        
        A = np.copy(background)  
        
        # Combine stereo views into a single composite image
        # Overlay left camera (red edges + texture) where it differs from background
        left_mask = np.any(A_left != background, axis=2)
        A[left_mask] = A_left[left_mask]
        
        # Overlay right camera (blue edges + texture) where it differs from background
        right_mask = np.any(A_right != background, axis=2)
        A[right_mask] = A_right[right_mask]


        # Display the rendered frame
        cv2.imshow("3D Computer Vision Demo", A)
        
        # Control animation speed
        # Uncomment cv2.waitKey(0) for frame-by-frame viewing
        # Default waits 1ms for smooth animation
        cv2.waitKey(1)







if __name__ == "__main__":
    main()
