from __future__ import print_function
import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter

def extract_features(imgs, feature_fns, verbose=False):
    num_imgs = imgs.shape[0]
    
    if num_imgs == 0:
        return np.array([])
    
    feature_dims = []
    first_img_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_img_features.append(feats)
        
    total_feature_dims = np.sum(feature_dims)
    imgs_features = np.zeros((num_imgs, total_feature_dims))
    imgs_features[0] = np.hstack(first_img_features).T
    
    for i in range(1, num_imgs):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
            
    if verbose and i % 1000 == 0:
        print('Done extracting features for %d / %d images' % (i, num_images))
        
    return imgs_features

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
def hog_feature(im):
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)
        
    sx, sy = image.shape
    orientations = 9
    cx, cy = (8, 8)
    
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90
    
    n_cellsx = int(np.floor(sx / cx))
    n_cellsy = int(np.floor(sy / cy))

    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)

        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx/2)::cx, int(cy/2)::cy].T

    return orientation_histogram.ravel()    
    
def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin+1)
    hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    return imhist