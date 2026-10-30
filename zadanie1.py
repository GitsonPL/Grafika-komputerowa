import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(image_url, headers=headers)

with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Obraz poczatkowy")
plt.axis("off")
plt.show()

width = int(img.shape[1] * 0.5)
height = int(img.shape[0] * 0.5)
dim = (width, height)

resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

rotated_img = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)

plt.figure(figsize=(6, 6))
plt.imshow(rotated_img, cmap="gray")
plt.title("Obraz po zmianach")
plt.axis("off")
plt.show()

print("Macierz obrazu:")
print(rotated_img[:5, :5])
print(f"\nKształt macierzy wynikowej: {rotated_img.shape}")