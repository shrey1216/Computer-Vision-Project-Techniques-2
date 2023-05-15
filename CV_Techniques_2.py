"""
1. Testing the functionality of image segmentation techniques and comparing
their accuracy requires us to work with images for which the correct
segmentation is known. Develop some test images by:

(a) Create an image SEG1 containing artificial objects on a background of
constant gray-level. Generate simple geometric objects such as squares,
rectangles, diamonds, stars, circles, etc., each having a constant gray-level
different from that of the background, some of them darker and some brighter
than the background. Determine the area of each object and store it in an
appropriate form.
"""

from google.colab.patches import cv2_imshow
from skimage.util import random_noise
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

#here is where we initialize SEG1
seg1 = np.zeros((500, 500, 3), dtype = "uint8")

#creating simple geometric objects
cv2.rectangle(seg1, (0, 0), (500, 500), (79, 79, 79), -1)
cv2.circle(seg1, (150, 120), 100, (135, 135, 135), -1)
cv2.rectangle(seg1, (250, 250), (480, 350), (28, 28, 28), -1)
cv2.rectangle(seg1, (50, 350), (150, 450), (220, 220, 220), -1)


#calculating the area of the circle, rectangle, and square using their coordinates
circleArea = math.pi * (100) * (100) 
rectangleArea = 230 * 100
squareArea = 100 * 100


cv2_imshow(seg1)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""(b) Superimpose additive Gaussian noise with a given standard deviation, thus
creating an image SEG2.
"""

# Add gaussian noise to the image.
seg2 = random_noise(seg1, mode='gaussian',seed=None,clip=True)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
seg2 = np.array(255*seg2, dtype = 'uint8')

# Display the noise image
cv2_imshow(seg2)

"""(c) Superimpose random impulse noise of a given severity over the image SEG2,
thus creating an image SEG3.

Salt and Pepper noise is used, as it is a type of impulse noise. 
"""

# Add salt-and-pepper noise to the image.
seg3 = random_noise(seg2, mode='s&p',seed=None,clip=True)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
seg3 = np.array(255*seg3, dtype = 'uint8')

# Display the noise image
cv2_imshow(seg3)

"""By varying the shapes of the objects, standard deviation of the Gaussian additive
noise, and severity of the impulse noise, sets of controlled properties can be
generated. To create a simple set of images for segmentation experiments, make
a single image SEG1, apply three levels of Gaussian additive noise, and three
levels of impulse noise. You will obtain a set of ten images that will be used in the
segmentation problems below. (20 points)
"""

#set: SEG1, SEG2, SEG3 from a, b, and c

#SEG4, SEG5, SEG6 (three levels of gaussian noise on SEG1) 
seg4 = random_noise(seg1, mode='gaussian',seed=None,clip=True)
seg4 = np.array(150*seg4, dtype = 'uint8')

seg5 = random_noise(seg1, mode='gaussian',seed=None,clip=True)
seg5 = np.array(300*seg5, dtype = 'uint8')

seg6 = random_noise(seg1, mode='gaussian',seed=None,clip=True)
seg6 = np.array(450*seg6, dtype = 'uint8')

#SEG7, SEG8, SEG9 (three levels of impulse noise on SEG1)
seg7 = random_noise(seg1, mode='s&p',seed=None,clip=True)
seg7 = np.array(150*seg7, dtype = 'uint8')

seg8 = random_noise(seg1, mode='s&p',seed=None,clip=True)
seg8 = np.array(300*seg8, dtype = 'uint8')

seg9 = random_noise(seg1, mode='s&p',seed=None,clip=True)
seg9 = np.array(450*seg9, dtype = 'uint8')

#print out SEG4, SEG5, SEG6, SEG7, SEG8, and SEG9
cv2_imshow(seg4)
cv2_imshow(seg5)
cv2_imshow(seg6)
cv2_imshow(seg7)
cv2_imshow(seg8)
cv2_imshow(seg9)

"""2. To assess the correctness of a segmentation, a set of measures must be
developed to allow quantitative comparison among methods. Develop a program
for calculating the following two segmentation accuracy indices:

(a) "Relative signed area error" is expressed in percent and computed as:
"""

#circleArea = math.pi * (100) * (100) 
#rectangleArea = 230 * 100
#squareArea = 100 * 100

#the true area is the sum of the areas of the square, rectangle, and square times 9
#I chose 9 because there are 9 SEG images in total. 
trueArea = (circleArea + rectangleArea + squareArea) * 9

#this is where I make my method for calculating the relative signed error.
#num is equal to the relative signed area error formula


def rse(totalArea, segArea):
  num = ((totalArea - segArea) / totalArea) * 100
  return num

"""(b)"Labelling error" is defined as the ratio of the number of incorrectly labeled pixels (object pixels labeled as background as vice versa) and
the number of pixels of true objects
according to prior knowledge, and is
expressed as percent. (20 points)
"""

def LE(totalArea, backgroundArea):
  num = (backgroundArea/totalArea) * 100
  return num

"""3.Implement the following methods for segmentation and apply to the test
images created in Problem 1. For each method and each image, quantitatively
assess the segmentation accuracy using the indices developed in Problem 2.
Compare the segmentation accuracy for individual methods.

(a) Basic thresholding.
"""

#we read all of the test images created in problem 1. 
seg1 = cv2.imread('SEG1.png',0)
seg2 = cv2.imread('SEG2.png',0)
seg3 = cv2.imread('SEG3.png',0)
seg4 = cv2.imread('SEG4.png',0)
seg5 = cv2.imread('SEG5.png',0)
seg6 = cv2.imread('SEG6.png',0)
seg7 = cv2.imread('SEG7.png',0)
seg8 = cv2.imread('SEG8.png',0)
seg9 = cv2.imread('SEG9.png',0)


#smoothing filter
blurseg1 = cv2.blur(seg1,(5,5))
blurseg2 = cv2.blur(seg2,(5,5))
blurseg3 = cv2.blur(seg3,(5,5))
blurseg4 = cv2.blur(seg4,(5,5))
blurseg5 = cv2.blur(seg5,(5,5))
blurseg6 = cv2.blur(seg6,(5,5))
blurseg7 = cv2.blur(seg7,(5,5))
blurseg8 = cv2.blur(seg8,(5,5))
blurseg9 = cv2.blur(seg9,(5,5))


#threshold
#Find the thresholds manually. Adjust the treshold values so the geometric objects are the most
#well defined and visible
ret,threshseg1 = cv2.threshold(blurseg1,134,255,cv2.THRESH_BINARY)
ret,threshseg2 = cv2.threshold(blurseg2,95,255,cv2.THRESH_BINARY)
ret,threshseg3 = cv2.threshold(blurseg3,109,255,cv2.THRESH_BINARY)
ret,threshseg4 = cv2.threshold(blurseg4,53,255,cv2.THRESH_BINARY)
ret,threshseg5 = cv2.threshold(blurseg5,98,255,cv2.THRESH_BINARY)
ret,threshseg6 = cv2.threshold(blurseg6,141,255,cv2.THRESH_BINARY)
ret,threshseg7 = cv2.threshold(blurseg7,61,255,cv2.THRESH_BINARY)
ret,threshseg8 = cv2.threshold(blurseg8,92,255,cv2.THRESH_BINARY)
ret,threshseg9 = cv2.threshold(blurseg9,146,255,cv2.THRESH_BINARY)


#display all of the tresholded images 
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg1, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG1", fontsize=12)

ax[1].imshow(threshseg1, cmap="gray")
ax[1].set_axis_off()
title = f'Thresholded Image SEG1'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg2, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG2", fontsize=12)

ax[3].imshow(threshseg2, cmap="gray")
ax[3].set_axis_off()
title = f'Thresholded Image SEG2'
ax[3].set_title(title, fontsize=12)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg3, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG3", fontsize=12)

ax[1].imshow(threshseg3, cmap="gray")
ax[1].set_axis_off()
title = f'Thresholded Image SEG3'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg4, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG4", fontsize=12)

ax[3].imshow(threshseg4, cmap="gray")
ax[3].set_axis_off()
title = f'Thresholded Image SEG4'
ax[3].set_title(title, fontsize=12)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg5, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG5", fontsize=12)

ax[1].imshow(threshseg5, cmap="gray")
ax[1].set_axis_off()
title = f'Thresholded Image SEG5'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg6, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG6", fontsize=12)

ax[3].imshow(threshseg6, cmap="gray")
ax[3].set_axis_off()
title = f'Thresholded Image SEG6'
ax[3].set_title(title, fontsize=12)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg7, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG7", fontsize=12)

ax[1].imshow(threshseg7, cmap="gray")
ax[1].set_axis_off()
title = f'Thresholded Image SEG7'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg8, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG8", fontsize=12)

ax[3].imshow(threshseg8, cmap="gray")
ax[3].set_axis_off()
title = f'Thresholded Image SEG8'
ax[3].set_title(title, fontsize=12)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg9, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG9", fontsize=12)

ax[1].imshow(threshseg9, cmap="gray")
ax[1].set_axis_off()
title = f'Thresholded Image SEG9'
ax[1].set_title(title, fontsize=12)

plt.show()

"""Segmentation Accuracy"""

#circleArea = math.pi * (100) * (100) 
#rectangleArea = 230 * 100
#squareArea = 100 * 100


#9 circles are segmented
#7 squares are segmented
#2 rectangles are segmented
segArea1 = (circleArea * 9) + (squareArea * 7) + (rectangleArea * 2)

rse(trueArea,segArea1)

#0 mislabeled circles
#2 mislabeled squares
#7 mislabeled rectangles
backArea1 = (circleArea * 0) + (squareArea * 2) + (rectangleArea * 7)


LE(trueArea,backArea1)

"""We see that thresholding returns a 31.221% relative signed error.
The labelling error also has the same percent value, of 31.221%.

(b) Chan-Vese.
"""

from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from PIL import Image

#seg1 = seg1.sum(-1)
#coins = data.coins()

#we change the pictures we have to grayscake to nake them 2D
seg1Img = Image.open("SEG1.png")
seg1 = seg1Img.convert('L')   # 'L' stands for 'luminosity'
seg1 = np.asarray(seg1)

seg2Img = Image.open("SEG2.png")
seg2 = seg2Img.convert('L')   
seg2 = np.asarray(seg2)

seg3Img = Image.open("SEG3.png")
seg3 = seg3Img.convert('L')   
seg3 = np.asarray(seg3)

seg4Img = Image.open("SEG4.png")
seg4 = seg4Img.convert('L')   
seg4 = np.asarray(seg4)

seg5Img = Image.open("SEG5.png")
seg5 = seg5Img.convert('L')   
seg5 = np.asarray(seg5)

seg6Img = Image.open("SEG6.png")
seg6 = seg6Img.convert('L')   
seg6 = np.asarray(seg6)

seg7Img = Image.open("SEG7.png")
seg7 = seg7Img.convert('L')   
seg7 = np.asarray(seg7)

seg8Img = Image.open("SEG8.png")
seg8 = seg8Img.convert('L')   
seg8 = np.asarray(seg8)

seg9Img = Image.open("SEG9.png")
seg9 = seg9Img.convert('L')   
seg9 = np.asarray(seg9)

#Here we can change the parameters to see how they impact the results
cv1 = chan_vese(seg1, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv2 = chan_vese(seg2, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv3 = chan_vese(seg3, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv4 = chan_vese(seg4, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv5 = chan_vese(seg5, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv6 = chan_vese(seg6, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv7 = chan_vese(seg7, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv8 = chan_vese(seg8, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

cv9 = chan_vese(seg9, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg1, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG1", fontsize=12)

ax[1].imshow(cv1[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv1[2])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg2, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG2", fontsize=12)

ax[3].imshow(cv2[0], cmap="gray")
ax[3].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv2[2])} iterations'
ax[3].set_title(title, fontsize=12)


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg3, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG3", fontsize=12)

ax[1].imshow(cv3[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv3[2])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg4, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG4", fontsize=12)

ax[3].imshow(cv4[0], cmap="gray")
ax[3].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv4[2])} iterations'
ax[3].set_title(title, fontsize=12)


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg5, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG5", fontsize=12)

ax[1].imshow(cv5[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv5[2])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg6, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG6", fontsize=12)

ax[3].imshow(cv6[0], cmap="gray")
ax[3].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv6[2])} iterations'
ax[3].set_title(title, fontsize=12)


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg7, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG7", fontsize=12)

ax[1].imshow(cv7[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv7[2])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(seg8, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Original Image SEG8", fontsize=12)

ax[3].imshow(cv8[0], cmap="gray")
ax[3].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv8[2])} iterations'
ax[3].set_title(title, fontsize=12)


fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(seg9, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image SEG9", fontsize=12)

ax[1].imshow(cv9[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv9[2])} iterations'
ax[1].set_title(title, fontsize=12)



fig.tight_layout()
plt.show()

"""Segmentation Accuracy"""

#circleArea = math.pi * (100) * (100) 
#rectangleArea = 230 * 100
#squareArea = 100 * 100


#7 circles are segmented
#8 squares are segmented
#3 rectangles are segmented
segArea2 = (circleArea * 7) + (squareArea * 8) + (rectangleArea * 3)

rse(trueArea,segArea2)

#2 mislabeled circles
#1 mislabeled squares
#6 mislabeled rectangles
backArea2 = (circleArea * 2) + (squareArea * 1) + (rectangleArea * 6)


LE(trueArea,backArea2)

"""We see that Chan-vese returns a 36.366% relative signed error. The labelling error also has the same percent value, of 36.366%.
This is higher than that of thresholding. For signed error, the lower the percentage, the better. This means that thresholding was more effective at segmentation than Can-vese was in this scenario.

4. Look up the Hough transform for finding line segments in an image. Write a paragraph about the Hough transform, explaining how it works and its advantages and weaknesses. 

The Hough transform is a technique which is used to isolate particular features within an image. The algorithm was originally developed by Paul V.C. Hough in 1962 to recognize complex lines in photographs. It has since been improved upon to allow it to detect other shapes as well, such as circles, ellipses, triangles, and quadrilaterals of certain shapes. To detect lines using a Hough transform, the image should first have its edges detected. This can be done using the canny or laplacian edge detection algorithms (any other algorithm can also be chosen for this step). This is important because we need edges to input into the Hough transform. Next, a Hough space should be made, and edge points from our image need to be mapped onto it. The Hough space is a 2D plane for which the *θ* values are represented by the horizontal axis, and the *ρ* values are represented by the vertical axis (made by a 2D array). We are using polar coordinates instead of cartesian coordinates because doing so will eliminate the issue of the unbounded value of the slope *m* for vertical lines had we used the form *y = mx + b*. The new form is *ρ = x cos(θ) + y sin(θ)* where *ρ* is the length of the normal line and *θ* is the angle between the normal line and the horizontal axis. Each edge point from the edge image produces a cosine curve in the Hough space, so mapping all the edge points will generate a lot of cosine curves. Edge points that lay on the same line will have their corresponding cosine curves intersect with each other on the specific *(ρ, θ)* pair. This is essentially how the Hough transform works, as the algorithm detects lines by finding the coordinate pairs that have a number of intersections larger than a certain threshold which is set by the user. The number of lines detected is based on this threshold. One advantage of the Hough transform is that the pixels lying on one line do not need to be touching. This can be useful when trying to detect lines with short breaks in them due to noise, or when objects are partially hidden from view. One disadvantage of using the Hough transform, is that it can give misleading results when objects happen to be aligned by chance. In addition to this, detected lines are infinite lines described by their slope and intercept values, rather than finite lines with defined end points.

5.Take the image heart.jpg and apply the linear heat equation to smooth it.
Specifically, write a program which will do the following:

(a) Load the image. Call this I(x,y)
"""

#here, we load the I(x,y) to be used in the heat equation. It is an imagine of a heart
#with its chambers visible. 

heart = cv2.imread('Heart.pbm',0)

"""(b) Discretize the heat equation"""

print("Discretize Heat Equation")

plate_length = 50
max_iter_time = 750

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

# Initialize solution: the grid of u(k, i, j)
u = np.empty((max_iter_time, plate_length, plate_length))

# Initial condition everywhere inside the grid
u_initial = 0

# Boundary conditions
u_top = 100.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0

# Set the initial condition
u.fill(u_initial)

# Set the boundary conditions
u[:, (plate_length-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (plate_length-1):] = u_right

def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]

    return u

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

# Do the calculation here
u = calculate(u)

"""(c) Output the smoothed image with the following number of iterations:
n=10, n=100, n=1000
"""

face = heart

min = np.min(face.shape)
max = np.max(face.shape)
imag = np.pad(face.copy(), 1, mode='constant')


#a function responsible for the time step updates (increases in n)
def time_step_update(imag,j,k, dt = .1):
  lplcn = imag[j,k-1] + imag[j-1,k] + imag[j,k+1] + imag[j+1,k] - 4*imag[j,k]
  return imag[j,k] + dt*lplcn

#we can change the number of iterations here, for n=10, n=100, and n=1000. The image is 
#smoothed/blurred more as we increase n. 

#n = 10
cur_imag = imag.copy()
for L in range(0,10):
    next_img = np.zeros(imag.shape)    
    for k in range(1, imag.shape[0]-1):
        for j in range(1, imag.shape[1]-1):
            next_img[j,k] = time_step_update(cur_imag,j,k)
    cur_imag = next_img
        
plt.figure(figsize=(20,20))
plt.subplot(121)
plt.title('Original Image', fontsize=18)
plt.imshow(imag, cmap=plt.cm.gray)
plt.subplot(122)
plt.title('Image at n = 10',fontsize=18)
plt.imshow(cur_imag, cmap=plt.cm.gray)

#n = 100
cur_imag = imag.copy()
for L in range(0,100):
    next_img = np.zeros(imag.shape)    
    for k in range(1, imag.shape[0]-1):
        for j in range(1, imag.shape[1]-1):
            next_img[j,k] = time_step_update(cur_imag,j,k)
    cur_imag = next_img
        
plt.figure(figsize=(20,20))
plt.subplot(121)
plt.title('Original Image', fontsize=18)
plt.imshow(imag, cmap=plt.cm.gray)
plt.subplot(122)
plt.title('Image at n = 100',fontsize=18)
plt.imshow(cur_imag, cmap=plt.cm.gray)

#n = 1000
cur_imag = imag.copy()
for L in range(0,1000):
    next_img = np.zeros(imag.shape)    
    for k in range(1, imag.shape[0]-1):
        for j in range(1, imag.shape[1]-1):
            next_img[j,k] = time_step_update(cur_imag,j,k)
    cur_imag = next_img
        
plt.figure(figsize=(20,20))
plt.subplot(121)
plt.title('Original Image', fontsize=18)
plt.imshow(imag, cmap=plt.cm.gray)
plt.subplot(122)
plt.title('Image at n = 1000',fontsize=18)
plt.imshow(cur_imag, cmap=plt.cm.gray)
