# Ramon Morros - Universitat Politecnica de Catalunya (UPC) - 2019
# This Software is provided "as is", without warranty of any kind.

import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.signal import convolve2d
import queue
from scipy.ndimage import gaussian_filter
from skimage.morphology import thin
import cv2


def convTri(I,r):
    '''
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r, r+1, r:-1:1]/(r+1)^2, the 2D version is simply convolve2d(f,f')).))
    Boundary effects are handled as if the image were padded symmetrically 
    prior to performing the convolution.
    
    Args:
        I (int): Image. 2D numpy array 
        r (int): Half length of the filter
    Returns:
        float  : Filtered image . 2D Numpy array 
    '''
    # Adapted from the Matlab version from:
    #Structured Edge Detection Toolbox V3.0                       
    #Piotr Dollar (pdollar-at-gmail.com)
    f = signal.windows.triang(2*r+1)
    f = f / np.sum(f)
    J = np.pad(I, [r,r], 'symmetric')
    J = convolve2d(convolve2d(I.astype(float), [f], 'full', boundary='symm'), np.vstack(f), 'full', boundary='symm')
    return J[r:-r,r:-r]

def orientation_from_edges(E, smooth_contours=True, sigma = 1.0):
    '''
    Compute an orientation map directly from a edge image. 
    The resulting orientation vectors are supposed to be perpendicular 
    to the contour direction

    Args:
        E (int)                : Edge image. 2D numpy array 
        smooth_contours (bool) : Whether to apply 2D low-pass filtering to the resulting angles (default: True)
    Returns:
        float  : Orientation map. 2D numpy array 

    '''
    # Adapted and extended from the Matlab version from:
    # Structured Edge Detection Toolbox V3.0                       
    # Piotr Dollar (pdollar-at-gmail.com)
    [Oy,Ox]   = np.gradient(convTri(E,4))
    [Oxy,Oxx] = np.gradient(Ox)
    [Oyy,Oyx] = np.gradient(Oy)
    O = np.mod(np.arctan(Oyy * np.sign(-Oyx) / (Oxx+1e-5)),np.pi);

    if smooth_contours:
        O = smoothorient(O, sigma)
    
    return O

def nms (E, O, r, s, m, thresh, extra_thinning=True):
    '''
    Non maxima supression. Pixels that are not local maxima
    along the direction given by O are suppressed (set to zero).

    Args:  
        E (int)               : Edge image. 2D numpy array 
        O (float)             : Orientation map. 2D numpy array 
        r (int)               : Radius for nms supr
        s (int)               : Distance (in pixels) for supression of noisy estimates near boundaries
        m (float)             : multiplier for conservative supr
        thresh (int)          : Low threshold in double thresholding
        extra_thinning (bool) : Whether to apply skeleton-based extra-thinning step (default: True)

    Returns:
        int                   : Edge image (thinned). 2D numpy array 
    '''
  
    # Adapted and extended from the matlab version from:
    # Structured Edge Detection Toolbox V3.0                       
    # Piotr Dollar (pdollar-at-gmail.com)

    h = E.shape[0]
    w = E.shape[1]

    x = list(range(w))
    y = list(range(h))
    f = interpolate.interp2d(x, y, E, kind='linear')

    valid   = E[E>thresh]
    valid_O = O[E>thresh]
    ky,kx   = np.where(E > thresh) 
    Oc = np.cos(valid_O)
    Os = np.sin(valid_O)
    
    for ii, ee in enumerate(valid):
        ee = ee*m
        for dd in range(-r,r+1):
            if dd != 0:
                x = kx[ii]
                y = ky[ii]
                e0 = f(x+dd*Oc[ii], y+dd*Os[ii])
                if ee < e0:
                    E[y,x] = 0
                    break

    # suppress noisy edge estimates near boundaries
    s = w/2 if s>w/2 else s
    s = h/2 if s>h/2 else s

    Dm = np.pad (np.ones(((h-2*s), (w-2*s)), dtype=float), (s,s), 'linear_ramp')
    E = E * Dm

    # Taken from Peter Kovesi Matlab functions https://www.peterkovesi.com/matlabfns/
    # Finally thin the 'nonmaximally suppressed' image by pointwise
    # multiplying itself with a morphological skeletonization of itself.

    # I know it is oxymoronic to thin a nonmaximally supressed image but
    # fixes the multiple adjacent peaks that can arise 
    # skel = bwmorph(im>0,'thin',Inf);
    if extra_thinning:
        tE = thin(E > thresh)
        E = E*tE

    # Remove isolated pixels
    #kernel = np.ones((3,3), dtype=int) * -1
    #kernel[1,1]=1
    #E = cv2.morphologyEx(E, cv2.MORPH_HITMISS, kernel)

    # Alternative version, as OpenCV Hit-or-Miss implementation is boken
    # https://stackoverflow.com/questions/46143800/removing-isolated-pixels-using-opencv
    E_comp = cv2.bitwise_not(E)  # could just use 255-img
    k1 = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]], np.uint8)
    k2 = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], np.uint8)
    hm1 = cv2.morphologyEx(E, cv2.MORPH_ERODE, k1)
    hm2 = cv2.morphologyEx(E_comp, cv2.MORPH_ERODE, k2)
    hm  = cv2.bitwise_and(hm1, hm2)
    E[hm == 0] = 0

    return E


def double_thresholding (edges, low_thresh, high_thresh):
    '''
    Double thresholding. Pixels below low_threshold are set to zero. Pixels
    between low_threshold and high_threshold are set to 128. Pixels equal or
    aboce high threshold are set to 255.

    Args:  
        edges (int)       : Edge image. 2D numpy array 
        low_thresh (int)  : Low threshold. 
        high_thresh (int) : High threshold. 

    Returns:
        int  : Edge image (thinned). 2D numpy array 
    '''

    # Arbitrary
    strong = 255
    weak   = 128

    t2edges = np.zeros(edges.shape)
    t2edges[np.logical_and(edges < high_thresh, edges >= low_thresh)] = weak
    t2edges[edges >= high_thresh] = strong

    return t2edges

    
def track_contours(t2edges, orientation, extra_thinning=False):
    '''
    Track contours from a double thresholded contour image using the gradient orientation 

    Args:  
        t2edges (int)         : Double-thresholded Edge image. 2D numpy array 
        orientation (float)   : Orientation map. 2D numpy array 
        extra_thinning (bool) : Whether to apply skeleton-based extra-thinning step (default: True)

    Returns:
        int                   : Edge image (thinned). 2D numpy array 
    '''
    edges = t2edges.copy()
    
    # Arbitrary
    strong = 255
    weak   = 128

    # Gradient is in the range [0..pi]. We will use 4 bins
    bins = [0.0, np.pi/8, np.pi/8+np.pi/4, np.pi/8+2*np.pi/4, np.pi/8+3*np.pi/4, np.pi]
    qor  = np.digitize(orientation, bins)
    qor[qor==5] = 1
    qor = qor - 1

    q = queue.Queue()

    # Associates angle bin with locations of neighbor pixels
    # Angle 0 is pointing upwards, positive angles are clockwise (until pi)
    #LUT = ( ((-1, 0),(1, 0)), ((-1, 1),(1,-1)), ((0, 1),(0, -1)), ((-1,-1),(+1,+1)) )
    LUT = ( ((-1, 0),(1, 0),(-1,-1),(+1,-1),(1,1),(1,-1)), ((-1, 1),(1,-1),(0,1),(-1,0),(0,-1),(1,0)), ((0, 1),(0, -1),(-1,1),(1,1),(-1,-1),(1,-1)), ((-1,-1),(+1,+1),(-1,0),(0,1),(1,0),(0,-1)) )
    
    # Add to the queue the weak contour point candidates near a strong contour point
    c_strong = np.transpose(np.where(edges == strong))
    # Convert to tuple to use the points to index arrays
    c_strong = tuple(map(tuple, c_strong))
    
    for cp in c_strong:
        # Search other candidate cp's along the contour direction and add them to the queue
        for off in LUT[qor[cp]]:
            nei = tuple(np.array(cp)+off)
            if edges[nei] == weak:
                q.put(nei)
                break;  # ???

    while q.empty() == False:
        # Retrieve a candidate from the queue
        cp = q.get()
        # Label as strong CP
        edges[cp] = strong

        # Search other candidate cp's along the contour direction and add to the queue
        for off in LUT[qor[cp]]:
            nei = tuple(np.array(cp)+off)
            if edges[nei] == weak:
                q.put(nei)
                break;

    # Remove the remaining weak edges
    edges[edges==weak] = 0

    if extra_thinning:
        tE = thin(edges != 0)
        edges = edges*tE
    
    return edges
        


def smoothorient(orient, sigma = 1.0):
    '''
    Smooth the orientation maps by applying low-pass filtering.

    Args:  
        orient (float) : Orientation map. 2D numpy array 
        sigma (float)  : Sigma parameter for the gaussian low pass filter

    Returns:
        int                   : Edge image (thinned). 2D numpy array 
    '''

    # Adapted from MATLAB and Octave Functions for Computer Vision and Image Processing
    # Peter Kovesi. https://www.peterkovesi.com/matlabfns/
    
    # Smoothing is applied separately to the sine and cosine of the angles to
    # avoid wraparound problems.
    cosor = np.cos(orient)
    sinor = np.sin(orient)

    cosor = gaussian_filter(cosor, sigma)
    sinor = gaussian_filter(sinor, sigma)
    
    # Reconstitute angles 
    smor = np.arctan2(sinor, cosor)
                                        
    return smor


def thin_contours (E, low_threshold, high_threshold, sigma=1.0):
    '''
    Thinning of edge map image. Applies non-maxima-suppression, 
    double thresholding (histeresys) and contour tracking.

    Args:  
        E (int)              : Edge image. 2D numpy array 
        low_threshold (int)  : Low threshold. 
        high_threshold (int) : High threshold. 

    Returns:
        int                   : Edge image (thinned). 2D numpy array 
    '''

    O = orientation_from_edges(E, sigma)
    Enms = nms(E,O, 1, 5, 1.01, low_threshold)
    Et = double_thresholding(Enms, low_threshold, high_threshold)
    Ef = track_contours(Et,O)

    return Ef
