import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#################
### 2D - PART ###
#################

def turn_r2(object2, gradus):
    radians = math.radians(gradus)
    rotation_matrix = np.array([
        [math.cos(radians), -math.sin(radians)],
        [math.sin(radians), math.cos(radians)]
    ])
    for i in range(len(object2)):
        object2[i] = np.dot(object2[i], rotation_matrix)
    return object2

def turn_r2_opencv_object(object2, angle):
    rotation_matrix = cv.getRotationMatrix2D((0, 0), angle, 1.0)
    rotated_object = cv.transform(object2.reshape(-1, 1, 2), rotation_matrix)
    return rotated_object.squeeze()

def turn_r2_opencv_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv.warpAffine(img, rotation_matrix, (w, h))
    return rotated_img

def scaling_r2(object2, kef):
    scaling_matrix = np.array([
        [kef, 0],
        [0, kef]
    ])
    for i in range(len(object2)):
        object2[i] = np.dot(object2[i], scaling_matrix)
    return object2

def scaling_cv_object(object2, scale_factor):
    scaling_matrix = np.array([
        [scale_factor, 0],
        [0, scale_factor]
    ])
    scaled_object = cv.transform(object2.reshape(-1, 1, 2), scaling_matrix)
    return scaled_object.squeeze()

def scaling_cv_image(img, scale_factor):
    res = cv.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
    return res


def mirroring_r2(object2, axis):
    if axis == "y":
        mirroring_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
        for i in range(len(object2)):
            object2[i] = np.dot(object2[i], mirroring_matrix)
    if axis == "x":
        mirroring_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])
        for i in range(len(object2)):
            object2[i] = np.dot(object2[i], mirroring_matrix)
    return object2

def mirroring_r2_cv_object(object2, axis):
    if axis == "y":
        mirroring_matrix = np.array([
            [1, 0],
            [0, -1]
        ], dtype=np.float32)
        object2 = cv.transform(object2.reshape(-1, 1, 2), mirroring_matrix)
    if axis == "x":
        mirroring_matrix = np.array([
            [-1, 0],
            [0, 1]
        ], dtype=np.float32)
        object2 = cv.transform(object2.reshape(-1, 1, 2), mirroring_matrix)
    return object2.squeeze()

def mirroring_r2_cv_image(img, axis):
    if axis == 'x':
        mirrored_img = cv.flip(img, 0)
    elif axis == 'y':
        mirrored_img = cv.flip(img, 1)
    else:
        mirrored_img = img
    return mirrored_img

def plot_with_axes_r2(original_object, transformed_object,  title):
    print("matrix",'\n', transformed_object)
    plt.plot(original_object[:, 0], original_object[:, 1], 'bo-', label='Початковий об\'єкт')
    # Змінений об'єкт
    plt.plot(transformed_object[:, 0], transformed_object[:, 1], 'ro-', label='Змінений об\'єкт')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title(title)
    plt.axis('equal')
    plt.legend()
    plt.show()

def universal_func(my_object, matrix):
    for i in range(len(my_object)):
        my_object[i] = np.dot(my_object[i], matrix)
    return my_object

def shear_r2(object2, axis, shear_factor):
    if axis == "x":
        shear_matrix = np.array([
            [1, shear_factor],
            [0, 1]
        ])
    elif axis == "y":
        shear_matrix = np.array([
            [1, 0],
            [shear_factor, 1]
        ])
    for i in range(len(object2)):
        object2[i] = np.dot(object2[i], shear_matrix)
    return object2

def shear_r2_cv_object(object2, axis, shear_factor):
    if axis == 'x':
        shear_matrix = np.array([
            [1, shear_factor, 0],
            [0, 1, 0]
        ], dtype=np.float32)
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0, 0],
            [shear_factor, 1, 0]
        ], dtype=np.float32)
    sheared_object = cv.transform(object2.reshape(-1, 1, 2), shear_matrix)
    return sheared_object.squeeze()

#################
### 3d - PART ###
#################

def turn_r3(object2, axis, gradus):
    radians = math.radians(gradus)
    if axis == "x":
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(radians), -math.sin(radians)],
            [0, math.sin(radians), math.cos(radians)]
        ])
    elif axis == "y":
        rotation_matrix = np.array([
            [math.cos(radians), 0, math.sin(radians)],
            [0, 1, 0],
            [-math.sin(radians), 0, math.cos(radians)]
        ])
    elif axis == "z":
        rotation_matrix = np.array([
            [math.cos(radians), -math.sin(radians), 0],
            [math.sin(radians), math.cos(radians), 0],
            [0, 0, 1]
        ])
    for i in range(len(object2)):
        object2[i] = np.dot(object2[i], rotation_matrix)
    return object2

def scaling_3d(object3, scale_factor):
    scaling_matrix = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, scale_factor]
    ])
    for i in range(len(object3)):
        object3[i] = np.dot(object3[i], scaling_matrix)
    return object3

def plot_with_edges_3d(my_object, title):
    print("matrix","\n",my_object)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(my_object)):
        for j in range(i, len(my_object)):
            ax.plot3D(*zip(my_object[i], my_object[j]), color='blue')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)
    plt.show()

# 2D transformation examples
object1 = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])

object2 = np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]])
original_object = object2.copy()
#Example 1: Rotation
byCV = turn_r2_opencv_object(object2, 45)
plot_with_axes_r2(original_object,byCV, "Rotation by OpenCV")
byTurn = turn_r2(object2, 45)
plot_with_axes_r2(original_object, byTurn, "Rotation by Custom Function")

#
# # # Example 2: Scaling
object2 = np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]])
byCV = scaling_cv_object(object2, 2)
plot_with_axes_r2(original_object,byCV, "Scaling by OpenCV")
byTurn = scaling_r2(object2, 2)
plot_with_axes_r2(original_object,byTurn, "Scaling by Custom Function")
# #
# # Example 3: Mirroring
object2 = np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]])
byCV = mirroring_r2_cv_object(object2, "y")
plot_with_axes_r2(original_object,byCV, "Mirroring by OpenCV")
byTurn = mirroring_r2(object2, "y")
plot_with_axes_r2(original_object,byTurn, "Mirroring by Custom Function")
#
# # Example 4: Shear
object2 = np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]])
byCV = shear_r2_cv_object(object2, "x", 0.5)
plot_with_axes_r2(original_object,byCV, "Shear by OpenCV")
byME = shear_r2(object2, "x", 0.5)
plot_with_axes_r2(original_object,byME, "Shear by Custom Function")

# # Example 5: Custom func
object2 = np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1], [0.5, 0.5]])
matrix = ([0,-5],[10,-3])
byME = universal_func(object2, matrix)
plot_with_axes_r2(original_object,byME, "Custom Function")

# # 3D transformation

object3_r3 = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1]
])

object3_r3 = turn_r3(object3_r3,"x",45)
plot_with_edges_3d(object3_r3,'turn by x')
object3_r3 = turn_r3(object3_r3,"y",45)
plot_with_edges_3d(object3_r3,'turn by y')
object3_r3 = turn_r3(object3_r3,"z",45)
plot_with_edges_3d(object3_r3,'turn by z')
object3_r3 = scaling_3d(object3_r3,5)
plot_with_edges_3d(object3_r3,'scaling')
# # 2D transformation on image
# # img_path = 'C:\\Users\\nekit\\Downloads\\1652231387_3-kartinkin-net-p-kartinki-kvadrata-4.jpg'
img_path = 'C:\\Users\\nekit\\Downloads\\kartinka-dobrogo-mirnogo-ranku.png'
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not read the image.")
else:
    ## Scaling the image
    dst = scaling_cv_image(img, 2)
    # Display the results
    cv.imshow('Original Image', img)
    cv.imshow('Scaled Image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Turn the image
    dst = turn_r2_opencv_image(img, 45)
    # Display the results
    cv.imshow('Original Image', img)
    cv.imshow('Turned Image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Mirror the image
    dst = mirroring_r2_cv_image(img, "y")
    # Display the results
    cv.imshow('Original Image', img)
    cv.imshow('Mirrored Image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Scale, turn and mirror the image
    dst = turn_r2_opencv_image(img, 45)
    dst = scaling_cv_image(dst, 0.5)
    dst = mirroring_r2_cv_image(dst, "x")
    # Display the results
    cv.imshow('Original Image', img)
    cv.imshow('Custom Image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


