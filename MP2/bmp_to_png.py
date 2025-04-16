import cv2
import os

ending = '.bmp'

# Find all files in the current directory that end with '.bmp'
files = [f for f in os.listdir('.') if f.endswith(ending)]

# Convert each one to a PNG or JPG file
for file in files:
    # Example file: 'gun_A_boundary.bmp'
    if not file.__contains__('_'):
        continue
    imagename = file.split('_')[0]  # Extract the image name
    # Read the BMP file
    img = cv2.imread(file)

    # Create a new filename by replacing the ending with '.png'
    new_file = file.replace(ending, '.png')
    # Write PNG files to folders based on kernel type
    output_folder = os.path.join('output', imagename)
    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Create folder for the kernel type
    kernel_name = file.split('_')[1]  # Extract the kernel name
    kernel_folder = os.path.join(output_folder, kernel_name)
    # Create the directory if it doesn't exist
    os.makedirs(kernel_folder, exist_ok=True)
    new_file = os.path.join(kernel_folder, new_file)
    # Save the image as a PNG file
    cv2.imwrite(new_file, img)

    # Optionally, you can also save it as JPG by changing the extension to '.jpg'
    # new_file_jpg = file.replace(ending, '.jpg')
    # cv2.imwrite(new_file_jpg, img)  # Uncomment this line to save as JPG
    print(f"Converted {file} to {new_file}")
