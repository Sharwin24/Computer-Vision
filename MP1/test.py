import cv2

# Display the image
img = cv2.imread('test.bmp', cv2.IMREAD_UNCHANGED)
if img is None:
    print("Error: Could not read the image.")

# Display image properties
print("Image size:", img.size)
print("Image dtype:", img.dtype)
print("Image channels:", img.shape[2] if len(img.shape) == 3 else 1)
print("Image width:", img.shape[1])
print("Image height:", img.shape[0])
print("Num Pixels: ", img.size // img.itemsize)
print("Bit Depth: ", img.itemsize * 8)


# Dump raw pixel values into log.txt
pixels = img.flatten()
with open('log.txt', 'w') as f:
    for i in range(len(pixels)):
        f.write(f'{pixels[i]} ')
        if (i + 1) % img.shape[1] == 0:
            f.write('\n')
