#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# In[2]:


test_img =  mpimg.imread("test2.jpg") 
im = mpimg.imread("test2.jpg") 
plt.imshow(im)


# In[3]:


def plt_images(orig_image, orig_title, processed_image, processed_title, cmap='gray'):
    # Visualize undirstorsion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    ax1.set_title(orig_title, fontsize=30)
    ax1.imshow(orig_image)
    ax2.set_title(processed_title, fontsize=30)
    ax2.imshow(processed_image, cmap='gray')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# In[4]:
src = np.float32([(550, 460),     # top-left
                               (150, 720),     # bottom-left
                               (1200, 720),    # bottom-right
                               (770, 460)])    # top-right
dst = np.float32([(100, 0),
                               (100, 720),
                               (1100, 720),
                               (1100, 0)])
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)
        
forward_image= cv2.warpPerspective(test_img, M, (1280, 720), flags=cv2.INTER_LINEAR)  
backward_image = cv2.warpPerspective(test_img, M_inv, (1280, 720), flags=cv2.INTER_LINEAR)
#prespective_forward(np.array(test_img),img_size= (1280,720),flags=cv2.INTER_LINEAR)        
plt_images(test_img, 'Source image',forward_image, 'bird eye view')


# # Thresholding
# 

# In[5]:


def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255



def threshold_forward(img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        v_channel = hsv[:,:,2]

        right_lane = threshold_rel(l_channel, 0.8, 1.0)
        right_lane[:,:750] = 0

        left_lane = threshold_abs(h_channel, 20, 30)
        left_lane &= threshold_rel(v_channel, 0.7, 1.0)
        left_lane[:,550:] = 0

        img2 = left_lane | right_lane

        return img2
thresholded_image = threshold_forward(forward_image)
plt_images(forward_image, 'Source image',thresholded_image, 'thresholded image')


# # Lane Line Detection

# In[6]:


left_fit = None
right_fit = None
binary = None
nonzero = None
nonzerox = None
nonzeroy = None
clear_visibility = True
dir = []
left_curve_img = mpimg.imread('left_turn.png')
right_curve_img = mpimg.imread('right_turn.png')
keep_straight_img = mpimg.imread('straight.png')
left_curve_img = cv2.normalize(src=left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
right_curve_img = cv2.normalize(src=right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
keep_straight_img = cv2.normalize(src=keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# HYPERPARAMETERS

# Number of sliding windows
nwindows = 9
# Width of the the windows +/- margin ,, width = 200
margin = 100
# Mininum number of pixels found to recenter window
minpix = 50

window_height=None
left_fit=None
right_fit=None


# In[7]:


def extract_features(img):
    """ Extract features from a binary image

    Parameters:
        img (np.array): A binary image
    """
    img = img
    # Height of of windows - based on nwindows and image shape
    global window_height
    window_height = np.int(img.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixel in the image
    global nonzero
    nonzero = img.nonzero()
    global nonzerox
    nonzerox = np.array(nonzero[1])
    global nonzeroy
    nonzeroy = np.array(nonzero[0])


# In[8]:


def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)


# In[9]:


##3 betkhaly el image 3 channels + bt return all pixels that belong to the lane
def find_lane_pixels(img):
    """Find lane pixels from a binary warped image.

    Parameters:
        img (np.array): A binary warped image

    Returns:
        leftx (np.array): x coordinates of left lane pixels
        lefty (np.array): y coordinates of left lane pixels
        rightx (np.array): x coordinates of right lane pixels
        righty (np.array): y coordinates of right lane pixels
        out_img (np.array): A RGB image that use to display result later on.
    """
    assert(len(img.shape) == 2)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))

    ## returns array of (x,sum of nonzero pixels)
    histogram = hist(img)
    midpoint = histogram.shape[0]//2
    ## x positions of max sum of left and right side of histogram
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Current position to be update later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    ## img shape[0] is the rows == y axis (vertical) bel tool 3shan ba3d keda yena2s menha fa yewsal le center eli 3aleha el door.
    y_current = img.shape[0] + window_height//2

    # Create empty lists to reveice left and right lane pixel
    leftx, lefty, rightx, righty = [], [], [], []

    # Step through the windows one by one
    ##  _ 3shan man3mlsh variable manstakhdmosh 3shan ehna nas professional
    for _ in range(nwindows):
        y_current -= window_height
        center_left = (leftx_current, y_current)
        center_right = (rightx_current, y_current)

        good_left_x, good_left_y = pixels_in_window(center_left, margin, window_height)
        good_right_x, good_right_y = pixels_in_window(center_right, margin, window_height)

        # Append these indices to the lists
        leftx.extend(good_left_x)
        lefty.extend(good_left_y)
        rightx.extend(good_right_x)
        righty.extend(good_right_y)

        ## recenter the window update x value 
        if len(good_left_x) > minpix:
            leftx_current = np.int32(np.mean(good_left_x))
        if len(good_right_x) > minpix:
            rightx_current = np.int32(np.mean(good_right_x))

    return leftx, lefty, rightx, righty, out_img


# In[10]:


##4 be7aded el two lines wel area el benhom betlwenha akhdar
def fit_poly(img):
    global left_fit
    global right_fit
    

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)


    ## polyfit(x,y,degree)
    if len(lefty) > 1500:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(righty) > 1500:
        right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting

    #akher el sora
    maxy = img.shape[0] - 1

    #lehad akher el taree2 ta2reban
    miny = img.shape[0] // 3


    #nezabat el min yeb2a min el no2at eli 3l taree2
    if len(lefty):
        
        miny = min(miny, np.min(lefty))

    if len(righty):
      
        miny = min(miny, np.min(righty))


    ## get x values to substitute in the poly fn.
    ploty = np.linspace(miny, maxy, img.shape[0])


    ##  x =  ay^2 + by+ c
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    ## byrsem el area el been el lanes bl akhdar
    # Visualization
    for i, y in enumerate(ploty):
        l = int(left_fitx[i])
        r = int(right_fitx[i])
        y = int(y)
        cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

  
  
    return out_img


# In[11]:


def lane_line_forward(img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        ##bamla el nonzero matrix
        extract_features(img)
        
        ##return el area el khadra
        return fit_poly(img)


# In[12]:


def pixels_in_window(center, margin, height):
        """ Return all pixel that in a specific window

        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window

        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)
        #check if the pixel is in the given window
        condx = (topleft[0] <= nonzerox) & (nonzerox <= bottomright[0])
        condy = (topleft[1] <= nonzeroy) & (nonzeroy <= bottomright[1])
        #return these pixels
        return nonzerox[condx&condy], nonzeroy[condx&condy]


# In[13]:


def measure_curvature():
        
        ##convert from pixels to meter
        ym = 30/720
        xm = 3.7/700

        left_fitt = left_fit.copy()
        right_fitt = right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR =  ((1 + (2*left_fitt[0] *y_eval + left_fitt[1])**2)**1.5)  / np.absolute(2*left_fitt[0])
        right_curveR = ((1 + (2*right_fitt[0]*y_eval + right_fitt[1])**2)**1.5) / np.absolute(2*right_fitt[0])

        xl = np.dot(left_fit, [700**2, 700, 1])
        xr = np.dot(right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm
        return left_curveR, right_curveR, pos 


# In[14]:


##5 detect L F R then draw them in a widget on the img
def plot(out_img):
    global dir
    global left_curve_img
    global right_curve_img
    global keep_straight_img
    np.set_printoptions(precision=6, suppress=True)
    lR, rR, pos = measure_curvature()

    value = None
    if abs(left_fit[0]) > abs(right_fit[0]):
        value = left_fit[0]
    else:
        value = right_fit[0]

        ## law el curvature olayla kamel odaamak , law kebera bl negative lef shemaal, law kebera + lef yemen
    if abs(value) <= 0.00015:
        dir.append('F')
    elif value < 0:
        dir.append('L')
    else:
        dir.append('R')

    if len(dir) > 10:
        dir.pop(0)

    W = 400
    H = 500
    widget = np.copy(out_img[:H, :W])
    widget //= 2
    widget[0,:] = [0, 0, 255]
    widget[-1,:] = [0, 0, 255]
    widget[:,0] = [0, 0, 255]
    widget[:,-1] = [0, 0, 255]
    out_img[:H, :W] = widget

    direction = max(set(dir), key = dir.count)
    msg = "Keep Straight Ahead"
    curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
    if direction == 'L':
        y, x = left_curve_img[:,:,3].nonzero()
        out_img[y, x-100+W//2] = left_curve_img[y, x, :3]
        msg = "Left Curve Ahead"
    if direction == 'R':
        y, x = right_curve_img[:,:,3].nonzero()
        out_img[y, x-100+W//2] = right_curve_img[y, x, :3]
        msg = "Right Curve Ahead"
    if direction == 'F':
        y, x = keep_straight_img[:,:,3].nonzero()
        out_img[y, x-100+W//2] = keep_straight_img[y, x, :3]

    cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    if direction in 'LR':
        cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.putText(
        out_img,
        "Good Lane Keeping",
        org=(10, 400),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=(0, 255, 0),
        thickness=2)

    cv2.putText(
        out_img,
        "Vehicle is {:.2f} m away from center".format(pos),
        org=(10, 450),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.66,
        color=(255, 255, 255),
        thickness=2)

    return out_img


# In[15]:


sora= cv2.warpPerspective(test_img, M, (1280, 720), flags=cv2.INTER_LINEAR)
sora=  threshold_forward(sora)
sora=lane_line_forward(sora)
sora = cv2.warpPerspective(sora, M_inv, (1280, 720), flags=cv2.INTER_LINEAR)

out_img = np.copy(test_img)
out_img = cv2.addWeighted(out_img, 1, sora, 0.6, 0)
out_img = plot(out_img)
plt_images(thresholded_image, 'Source image',out_img, 'Output image')


# In[16]:


def main_forward(img):
  sora= cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_LINEAR)
  sora=  threshold_forward(sora)
  sora=lane_line_forward(sora)
  sora = cv2.warpPerspective(sora, M_inv, (1280, 720), flags=cv2.INTER_LINEAR)

  out_img = np.copy(img)
  out_img = cv2.addWeighted(out_img, 1, sora, 0.6, 0)
  out_img = plot(out_img)
  return out_img
  


# In[23]:


from moviepy.editor import VideoFileClip

import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
mode = sys.argv[3]
def process_video(input_path, output_path, mode):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(main_forward)
        out_clip.write_videofile(output_path, audio=False)
        if(mode !='1'): # debugging mode
          sora = clip.get_frame(1) #hna5od frame mn el video

          sora1 = cv2.warpPerspective(sora, M, (1280, 720), flags=cv2.INTER_LINEAR) # warp image 
          sora2 =  threshold_forward(sora1) # image with lane lines detected
          sora3 = lane_line_forward(sora2) # image with lanes detected
          sora4 = cv2.warpPerspective(sora3, M_inv, (1280, 720), flags=cv2.INTER_LINEAR) # el image b3d mareg3et l2slaha

          out_img = np.copy(sora)
          out_img = cv2.addWeighted(out_img, 1, sora4, 0.6, 0)
          opencv_rgb_img = cv2.cvtColor(sora2, cv2.COLOR_GRAY2RGB)
          out_img2 = np.concatenate((sora, sora1, opencv_rgb_img, sora3, sora4, out_img), axis=0)
          plt.figure(figsize = (20, 15))
          plt.imshow(out_img2)
          plt.savefig('foo.png')
        
process_video(input_path, output_path, mode)


# In[ ]:




