import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

print("Pobieranie zdjęcia...")
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(image_url, headers=headers)

with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Oryginalne zdjęcie")
plt.axis("off")
plt.show()

hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
color_channels = ('r', 'g', 'b')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_gray, color='black')
plt.title("Histogram - Całe zdjęcie")
plt.xlabel("Poziom jasności (0-255)")
plt.ylabel("Liczba pikseli")

plt.subplot(1, 2, 2)
for i, col in enumerate(color_channels):
    hist_channel = cv2.calcHist([img_rgb], [i], None, [256], [0, 256]).flatten()
    plt.plot(hist_channel, color=col)
plt.title("Histogram - Kanały RGB")
plt.xlabel("Poziom jasności (0-255)")
plt.ylabel("Liczba pikseli")

plt.tight_layout()
plt.show()

total_pixels = gray.size
dark_pixels_pct = (float(hist_gray[0]) / total_pixels) * 100
bright_pixels_pct = (float(hist_gray[255]) / total_pixels) * 100

non_zero_bins = np.where(hist_gray > 0)[0]
dynamic_range = non_zero_bins[-1] - non_zero_bins[0] if len(non_zero_bins) > 0 else 0

print("\n--- ANALIZA JAKOŚCI ZDJĘCIA ---")
print(f"Odsetek pikseli skrajnie ciemnych: {dark_pixels_pct:.2f}%")
print(f"Odsetek pikseli skrajnie jasnych: {bright_pixels_pct:.2f}%")
print(f"Rozpiętość tonalna: {dynamic_range} / 255")

quality_score = "Dobra"
reasons = []

if dark_pixels_pct > 15:
    quality_score = "Średnia / Niska"
    reasons.append("Zbyt dużo ciemnych obszarów.")
if bright_pixels_pct > 15:
    quality_score = "Średnia / Niska"
    reasons.append("Zbyt dużo jasnych obszarów (przepalenia).")
if dynamic_range < 150:
    quality_score = "Niska"
    reasons.append("Niska rozpiętość tonalna.")

print(f"\nOcena jakości: {quality_score}")
if reasons:
    print("Uwagi:", " ".join(reasons))
else:
    print("Uwagi: Zdjęcie posiada zbalansowany kontrast i poprawną ekspozycję.")