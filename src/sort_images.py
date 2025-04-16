# %%
import os

# List all images in the directory
path = 'processed_data/all_images'

# Create a list of all the image paths
image_paths = os.listdir(path)

print('Number of images: {}'.format(len(image_paths)))

# %%
import cv2

from PIL import Image
from IPython.display import display

# Get a random image index
import random
index = random.randint(0, len(image_paths)-1)

# Get the image name
image_name = image_paths[index]

# Load the image from a file path
img_path = os.path.join(path, image_name)
img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Bounding box (top, bottom, left, right)
box = (119.0, 187.0, 381.0, 911.0)

# Crop the image
img_arr = img_arr[int(box[0]):int(box[1]), int(box[2]):int(box[3])]

# Convert the image to a PIL image
img = Image.fromarray(img_arr)

# Print number of pixels
print('Number of pixels: {}'.format(img.size[0] * img.size[1]))


# Show the image in Jupyter notebook
display(img)



# %%
# Print the first row of pixels in the image to contain a white pixel



def get_box(img_arr):
    img_height = img_arr.shape[0]
    img_width = img_arr.shape[1]


    top = 0
    bottom = 0
    left = 0
    right = 0

    # print('Image height: {}'.format(img_height))
    # print('Image width: {}'.format(img_width))

    for i in range(img_height):
        row = img_arr[i]
        # Check if the row contains a white pixel
        if 255 in row:
            # print('First row with white pixel: {}'.format(i))
            top = i
            break

    for i in range(img_height-1, -1, -1):
        row = img_arr[i]
        # Check if the row contains a white pixel
        if 255 in row:
            # print('Last row with white pixel: {}'.format(i))
            bottom = i
            break

    for i in range(img_width):
        col = img_arr[:, i]
        # Check if the column contains a white pixel
        if 255 in col:
            # print('First column with white pixel: {}'.format(i))
            left = i
            break

    for i in range(img_width-1, -1, -1):
        col = img_arr[:, i]
        # Check if the column contains a white pixel
        if 255 in col:
            # print('Last column with white pixel: {}'.format(i))
            right = i
            break

    return top, bottom, left, right

top, bottom, left, right = get_box(img_arr)

# Crop the image to the bounding box
img = img.crop((left, top, right, bottom))

# Show the image in Jupyter notebook
display(img)



# %%
# bounding_boxes = {}

# count = 0
# for image_name in image_paths:
#     img_path = os.path.join(path, image_name)
#     img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     top, bottom, left, right = get_box(img_arr)
#     bounding_boxes[image_name] = (top, bottom, left, right)

#     count += 1
#     if count % 1000 == 0:
#         print('Processed {} images'.format(count))

# print(bounding_boxes)


# import json

# # Save the bounding boxes to a JSON file
# with open('processed_data/bounding_boxes.json', 'w') as f:
#     json.dump(bounding_boxes, f)


# %%
import json

# Load the bounding boxes from a JSON file
bounding_boxes = json.load(open('processed_data/bounding_boxes.json'))

# %%
# Fine the mean, median, and standard deviation of each position on the bounding box
import numpy as np

tops = []
bottoms = []
lefts = []
rights = []

for image_name, (top, bottom, left, right) in bounding_boxes.items():
    tops.append(top)
    bottoms.append(bottom)
    lefts.append(left)
    rights.append(right)

print('Mean top: {}'.format(np.mean(tops)))
print('Median top: {}'.format(np.median(tops)))
print('Standard deviation top: {}'.format(np.std(tops)))
print('Highest top: {}'.format(np.min(tops)))
print('Lowest top: {}'.format(np.max(tops)))
print()

print('Mean bottom: {}'.format(np.mean(bottoms)))
print('Median bottom: {}'.format(np.median(bottoms)))
print('Standard deviation bottom: {}'.format(np.std(bottoms)))
print('Lowest bottom: {}'.format(np.max(bottoms)))
print('Highest bottom: {}'.format(np.min(bottoms)))
print()

print('Mean left: {}'.format(np.mean(lefts)))
print('Median left: {}'.format(np.median(lefts)))
print('Standard deviation left: {}'.format(np.std(lefts)))
print('Widest left: {}'.format(np.min(lefts)))
print('Narrowest left: {}'.format(np.max(lefts)))
print()

print('Mean right: {}'.format(np.mean(rights)))
print('Median right: {}'.format(np.median(rights)))
print('Standard deviation right: {}'.format(np.std(rights)))
print('Widest right: {}'.format(np.max(rights)))
print('Narrowest right: {}'.format(np.min(rights)))
print()




# %%
# Graph the distribution of the bounding box positions

import matplotlib.pyplot as plt

plt.hist(tops, bins=100)

plt.hist(bottoms, bins=100)

plt.title('Top/Bottom')

# %%
plt.hist(lefts, bins=100)

plt.hist(rights, bins=100)

plt.title('Left/Right')

# %%
import numpy as np

def find_x_percent_covering_bbox(bbox_dict, x=90):
    tops = []
    bottoms = []
    lefts = []
    rights = []

    # Collect all the values for each dimension in separate lists
    for bbox in bbox_dict.values():
        top, bottom, left, right = bbox
        tops.append(top)
        bottoms.append(bottom)
        lefts.append(left)
        rights.append(right)

    # Calculate the 5th and 95th percentiles for each dimension
    top_5th = np.percentile(tops, (100-x)/2)
    bottom_95th = np.percentile(bottoms, 100-(100-x)/2)
    left_5th = np.percentile(lefts, (100-x)/2)
    right_95th = np.percentile(rights, 100-(100-x)/2)

    # Create the 90% covering bounding box
    covering_bbox = (top_5th, bottom_95th, left_5th, right_95th)

    return covering_bbox

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=100)
print("100% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=95)
print("95% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=90)
print("90% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=80)
print("80% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=70)
print("70% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=60)
print("60% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=50)
print("50% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))

covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=5)
print("10% Covering:", covering_bbox)
print('Num pixels: {}'.format((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2])))
print()


pixels = []
for i in range(0, 101):
    covering_bbox = find_x_percent_covering_bbox(bounding_boxes, x=i)
    pixels.append((covering_bbox[1]-covering_bbox[0])*(covering_bbox[3]-covering_bbox[2]))

plt.plot(range(0, 101), pixels)
# add more ticks to the x-axis
plt.xticks(range(0, 101, 5))
plt.xlabel('Percent of Images')
plt.ylabel('Number of Pixels')

plt.title('Pixels in Bounding Box')



# %%
def filter_elements_inside_bbox(bbox_dict, target_bbox):
    target_top, target_bottom, target_left, target_right = target_bbox
    inside_bbox = {}

    for filename, bbox in bbox_dict.items():
        top, bottom, left, right = bbox

        if (top >= target_top and bottom <= target_bottom and
                left >= target_left and right <= target_right):
            inside_bbox[filename] = bbox

    return inside_bbox

# target_bbox = (118.0, 199.0, 298.0, 993.0)  # Replace this with your target bounding box
target_bbox = (118.0, 199.0, 298.0, 993.0)  # Replace this with your target bounding box
filtered_elements = filter_elements_inside_bbox(bounding_boxes, target_bbox)

print("Length of filtered elements:", len(filtered_elements))


# %%
import cv2

from PIL import Image
from IPython.display import display

# get random item from dictionary
image_name = random.choice(list(filtered_elements.keys()))

# Load the image from a file path
img_path = os.path.join(path, image_name)
img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Bounding box (top, bottom, left, right)
box = (118.0, 199.0, 298.0, 993.0)

# Crop the image
img_arr = img_arr[int(box[0]):int(box[1]), int(box[2]):int(box[3])]

print('Image shape: {}'.format(img_arr.shape))

# Convert the image to a PIL image
img = Image.fromarray(img_arr)


# Print number of pixels
print('Number of pixels: {}'.format(img.size[0] * img.size[1]))


# Show the image in Jupyter notebook
display(img)


# %%
# count = 0

# for image_name, bbox in filtered_elements.items():
#     # Load the image from a file path
#     img_path = os.path.join(path, image_name)
#     img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#     # Bounding box (top, bottom, left, right)
#     box = (118.0, 199.0, 298.0, 993.0)

#     # # Convert the box so the size is divisible by 4, but still covers the same area
#     # box = (box[0] - box[0] % 4, box[1] + (4 - box[1] % 4), box[2] - box[2] % 4, box[3] + (4 - box[3] % 4))

#     # Crop the image
#     img_arr = img_arr[int(box[0]):int(box[1]), int(box[2]):int(box[3])]

#     # Save the image to a file path
#     save_path = 'processed_data/small_images/' + image_name
#     cv2.imwrite(save_path, img_arr)

#     count += 1

#     if count % 1000 == 0:
#         print("Processed {} images".format(count))




# %%
import os

# open processed_data/all_formulas.txt
with open('processed_data/all_formulas.txt', 'r') as f:
    formulas = f.readlines()

# Print number of images in all_images
print("Number of images:", len(os.listdir('processed_data/all_images')))

print(len(formulas))

with open('processed_data/small_formulas.txt', 'w') as f:
    for formula in formulas:
        if formula.split(' ')[0] in filtered_elements:
            f.write(formula)

# Print the number of formulas in small_formulas.txt
print("Number of formulas:", len(open('processed_data/small_formulas.txt').readlines()))

# Print number of images in small_images
print("Number of images:", len(os.listdir('processed_data/small_images')))

# %%
# import os
# import random
# from PIL import Image

# # Set the path
# path = 'processed_data/validate_images/'

# # Get a list of all image files in the directory
# file_list = [file_name for file_name in os.listdir(path) if file_name.endswith('.png')]

# # Choose a random image from the list
# random_file = random.choice(file_list)

# # Load the image using PIL
# image = Image.open(os.path.join(path, random_file))

# # Get the image size as a tuple in the format (width, height)
# size = image.size

# # Print the shape of the image
# print(size)

# print()