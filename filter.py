# Convert the image to HSV color space
import cv2
import numpy as np

def HLS_filter(image, hls_params):
    # hls_params = hls_params.astype(np.uint8).flatten()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # Define the lower and upper bounds of HSV
    lower_bound = np.array([hls_params[0], hls_params[1], hls_params[2]])
    upper_bound = np.array([hls_params[3], hls_params[4], hls_params[5]])

    print(lower_bound,upper_bound)
    # Create a mask based on the lower and upper bounds
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask