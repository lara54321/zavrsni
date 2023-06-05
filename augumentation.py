import cv2
import os
import numpy as np
import random

# Function to flip image horizontally
def flip_image_horizontal(image):
    return cv2.flip(image, 1)  # Flips the image horizontally

# Function to flip image vertically
def flip_image_vertical(image):
    return cv2.flip(image, 0)  # Flips the image vertically

# Function to rotate image
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (cols, rows))

# Function to translate image
def translate_image(image, shift_x, shift_y):
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, matrix, (cols, rows))
    
    # Find non-zero pixels in the translated image
    non_zero_pixels = cv2.findNonZero(cv2.cvtColor(translated_image, cv2.COLOR_BGR2GRAY))
    
    # Get the bounding box coordinates of the non-zero pixels
    if non_zero_pixels is not None:
        (x, y, w, h) = cv2.boundingRect(non_zero_pixels)
        translated_image = translated_image[y:y+h, x:x+w]
    
    return translated_image

# Set the path to the directory containing the images
image_directory = 'images_512'
output_directory = 'augumentation_512'

# Get the list of image filenames in the directory
image_filenames = os.listdir(image_directory)

# Set the number of variations and maximum shift amounts
num_variations = 5

# Loop over each image file
for filename in image_filenames:
    # Construct the full path to the image file
    image_path = os.path.join(image_directory, filename)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Get the height and width of the image
    height, width = image.shape[:2]
    image_size = height
    max_shift = round(height * 0.4)
    
    # Apply data augmentation techniques
    flipped_horizontal_image = flip_image_horizontal(image)
    flipped_vertical_image = flip_image_vertical(image)
    #rotated_45_image = rotate_image(image, 45)
    rotated_90_image = rotate_image(image, 90)
    rotated_minus_90_image = rotate_image(image, -90)
    #rotated_minus_45_image = rotate_image(image, -45)
    rotated_180_image = rotate_image(image, 180)
    
    # Save the augmented images with appropriate filenames
    flipped_horizontal_filename = os.path.splitext(filename)[0] + '_flipped_horizontal.jpg'
    flipped_vertical_filename = os.path.splitext(filename)[0] + '_flipped_vertical.jpg'
    #rotated_45_filename = os.path.splitext(filename)[0] + '_rotated_45.jpg'
    rotated_90_filename = os.path.splitext(filename)[0] + '_rotated_90.jpg'
    rotated_minus_90_filename = os.path.splitext(filename)[0] + '_rotated_minus_90.jpg'
    #rotated_minus_45_filename = os.path.splitext(filename)[0] + '_rotated_minus_45.jpg'
    rotated_180_filename = os.path.splitext(filename)[0] + '_rotated_180.jpg'
    translated_filename = os.path.splitext(filename)[0] + '_translated.jpg'
    
    cv2.imwrite(os.path.join(output_directory, flipped_horizontal_filename), flipped_horizontal_image)
    cv2.imwrite(os.path.join(output_directory, flipped_vertical_filename), flipped_vertical_image)
    #cv2.imwrite(os.path.join(image_directory, rotated_45_filename), rotated_45_image)
    cv2.imwrite(os.path.join(output_directory, rotated_90_filename), rotated_90_image)
    cv2.imwrite(os.path.join(output_directory, rotated_minus_90_filename), rotated_minus_90_image)
    #cv2.imwrite(os.path.join(image_directory, rotated_minus_45_filename), rotated_minus_45_image)
    cv2.imwrite(os.path.join(output_directory, rotated_180_filename), rotated_180_image)
    
    # Apply variations on the translation
    for i in range(num_variations):
        # Generate random shift amounts within the maximum shift range
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = shift_x
        
        # Apply translation to the image
        translated_image = translate_image(image, shift_x, shift_y)
        translated_image = cv2.resize(translated_image, (width, height))
        
        # Save the augmented image with the appropriate filename
        translated_filename = os.path.splitext(filename)[0] + f'_translated_{i}.jpg'
        cv2.imwrite(os.path.join(output_directory, translated_filename), translated_image)

