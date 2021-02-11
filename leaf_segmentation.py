import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
import scipy
from skimage import io
from skimage import filters
from skimage import measure
from skimage import segmentation
from skimage import transform
from skimage.feature import canny
from skimage import util

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker

from matplotlib import collections  as mc
import math

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
tree_dir = os.path.join("leaf scans US_2020-07-21", "tree5")
img_name = "7_21_20_5_5t0001.jpg"

# prepare some plotting shenanigans
f, axarr = plt.subplots(2,4)

# full image path
filename = os.path.join(data_dir, tree_dir, img_name)

# load image into a array of numbers
cimg = io.imread(filename)

# pick only green component (R=0, G=1, B=2)
img = cimg[:, :, 1]
axarr[0,0].imshow(img, cmap='gray') 

# simple threashold mask to get leafs
val = filters.threshold_otsu(img)
mask = img < val
axarr[0,1].imshow(mask, cmap='gray') 

# blur mask to remove noise
mask_blurred = filters.gaussian(mask, 10)
axarr[0,2].imshow(mask_blurred, cmap='gray') 

# mask out blurred image
mask_fixed = mask_blurred > 0.99
axarr[0,3].imshow(mask_fixed, cmap='gray')

# label components
labels = measure.label(mask_fixed, background=0)
axarr[1, 0].imshow(labels)
properties = measure.regionprops(labels)

centroids_x = [p["centroid"][0] for p in properties]
centroids_y = [p["centroid"][1] for p in properties]

# functions `mask_...` take an image and produce 0-1 image, 0 - not a leaf, 1 - a leaf

def mask_blur(image, blur_amount = 10.0):

    val = filters.threshold_otsu(image)
    mask = img < val
    mask = filters.gaussian(mask, blur_amount)
    mask = mask > 0.99

    return mask

def mask_morphology(image, iterations = 10):

    val = filters.threshold_otsu(image)
    mask = img < val

    mask = scipy.ndimage.binary_opening(mask, iterations = min(5, iterations))
    mask = scipy.ndimage.binary_fill_holes(mask)
    mask = scipy.ndimage.binary_opening(mask, iterations = iterations)

    return mask

# takes a `mask` (i.e. 0-1 image of where leafs are) and produces image labeling 

# def segment_components(mask): 

# def watershed_segmentation(image):
#     distance = ndimage.distance_transform_edt(image)
#     local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
#     markers = measure.label(local_maxi)
#     labels_ws = watershed(-distance, markers, mask=image)
#     return distance

def swap_axes(x):
    return np.array([x[1],x[0]])


def get_axes(p):

    c = swap_axes(np.array(p["centroid"]))
    th = p["orientation"]
    d1 = swap_axes(np.array(( math.cos(th), math.sin(th))))
    d2 = swap_axes(np.array((-math.sin(th), math.cos(th))))

    l1 = p["major_axis_length"]
    l2 = p["minor_axis_length"]

    return [[c + 0.5*l1*d1, c - 0.5*l1*d1],
            [c + 0.5*l2*d2, c - 0.5*l2*d2]]


leaf_id = 2

lines = sum([get_axes(p) for p in properties],[])
lc = mc.LineCollection(lines)
    
axarr[0,0].add_collection(lc)
axarr[0,0].scatter(centroids_y, centroids_x, color = 'w', s = 10)


def get_leaf(image, prop):
    img_height = image.shape[0]
    img_width = image.shape[1]
    img_center = 0.5*np.array([img_width, img_height])
    
    leaf_center = swap_axes(np.array(prop["centroid"]))
    angle = - prop["orientation"] * 180.0/math.pi

    trans1 = transform.EuclideanTransform(translation = leaf_center - img_center)

    transformed_image = transform.warp(image, trans1, mode='constant')
    transformed_image = transform.rotate(transformed_image, angle = angle, center = img_center)

    major_axis = prop["major_axis_length"]
    minor_axis = prop["minor_axis_length"]

    # Final size of the image
    H = major_axis + 200
    W = minor_axis + 150

    # How much we have to crop original image to get (H x W) image
    top_crop = 0.5*(img_height - H)
    bottom_crop = top_crop
    left_crop = 0.5*(img_width - W)
    right_crop = left_crop

    crop_dims = [[top_crop, bottom_crop], [left_crop, right_crop]]

    # makes sure that the crop function works for color images too
    if len(image.shape)==3:
        crop_dims.append([0,0])

    transformed_image = util.crop(transformed_image, crop_dims)  

    return transformed_image


leaf_img = get_leaf(cimg, properties[leaf_id])

axarr[1,1].imshow(leaf_img, cmap='gray')
# axarr[1,2].imshow(watershed_segmentation(mask_fixed))
# axarr[1,3].imshow(skimage.filters.sobel(img), cmap='gray')
it = 5
axarr[1,2].imshow(mask_morphology(img, iterations = 20))

axarr[1,3].imshow(scipy.ndimage.binary_opening(scipy.ndimage.binary_closing(mask, iterations= it), iterations = it))


# Random walter segmentation
# markers[~image] = -1
# labels_rw = segmentation.random_walker(mask_fixed, markers)

plt.show()

