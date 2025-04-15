import cv2
import os

ending = '.bmp'

# Find all files in the current directory that end with '.bmp'
files = [f for f in os.listdir('.') if f.endswith(ending)]

# Convert each one to a PNG or JPG file
for file in files:
    # Read the BMP file
    img = cv2.imread(file)

    # Create a new filename by replacing the ending with '.png'
    new_file = file.replace(ending, '.png')

    # Save the image as a PNG file
    cv2.imwrite(new_file, img)

    # Optionally, you can also save it as JPG by changing the extension to '.jpg'
    # new_file_jpg = file.replace(ending, '.jpg')
    # cv2.imwrite(new_file_jpg, img)  # Uncomment this line to save as JPG
    print(f"Converted {file} to {new_file}")
