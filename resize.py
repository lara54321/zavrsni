from turtle import right
from PIL import Image
import math
import os


image_size = 512

def all_same_size(image):
    width, height = image.size
    center_x = width / 2
    center_y = height / 2

    left = center_x - image_size/2
    upper = center_y - image_size/2
    right = center_x + image_size/2
    bottom = center_y + image_size/2
    image = image.crop((left, upper, right, bottom))
    return image

def crop_image(image, label_path):
    # Open the image
    with open(label_path, 'r') as file:
    # Read the lines from the file
        lines = file.readlines()
        width, height = image.size
        cropped_images = []

    # Extract the values from the first line (assuming there is only one line in the file)
        for line in lines:
        # Extract the values from the line
            values = line.split()
            object_class = values[0]
            x = float(values[1]) * width
            y = float(values[2]) * height
            x_width = float(values[3]) * width
            y_height = float(values[4]) * height
            left = x - x_width/2 - 0.3 * width
            upper = y - y_height/2 - 0.3 * height
            right = x + x_width/2 + 0.3 * width
            bottom = y + y_height/2 + 0.3 * height

        
        # Crop the image
            cropped = image.crop((left, upper, right, bottom))
            cropped_images.append(cropped)
    
    return cropped_images

input_images = ('original/images')
output_folder = (f'images_{image_size}')
label_folder = ('labels')
images = os.listdir(input_images)
labels = os.listdir('labels')
for image_name in images:

    image_path = os.path.join(input_images, image_name)
    image = Image.open(image_path)
    output_path = os.path.join(output_folder, image_name)
    name = os.path.splitext(image_name)[0]
    label_name = name + '.txt'
    label_path = os.path.join(label_folder, label_name)
    print(image_name)
    print(label_name)

    cropped_images = crop_image(image, label_path)  
    i = 0
    for image in cropped_images:
    # Save the cropped image
        image_resized = all_same_size(image)
        unique_image_name = f"{name}_{i}.png"
        output_path = os.path.join(output_folder, unique_image_name)
        image_resized.save(output_path)
        i += 1