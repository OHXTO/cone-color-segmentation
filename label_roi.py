import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly

# picture path
img_dir = os.path.join(os.getcwd(), "ECE5242Proj1-train")


# get all file name
names = [n for n in os.listdir(img_dir) if n.endswith(".png")]
names.sort()

print("found", len(names),"first 3 png", names[:3])


# ROI mark each image and collect HSV.
# X_list element is pixels  = img_hsv[mask] array (Ni pixels, 3 channel represents H,S,V)

X_cone_list = []
X_bg_list = []

for index, n in enumerate(names, start=1):

    path = os.path.join(img_dir, n)         # full path with file name
    img = cv2.imread(path)                  # read image
    
    if img is None: 
        print("fail",path)
        continue

    # show in rgb but train by HSV. HSV is better for color segmentation. gray for drawing transfer to pixel
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # show window and draw Cone ROI
    plt.figure()
    plt.imshow(img_rgb)
    plt.title(f"{index}/{len(names)} \n draw CONE ROI double-click to finish")
    roi = RoiPoly(color='r')
    mask = roi.get_mask(gray) # get (Height, Width) bool from gray casue TA said
    plt.close()

    # Cone ROI, Save drawing from mask = roi.get_mask(img)
    cone_pixels = img_hsv[mask]  # (N,3) N pixels, and 3 channel H,S,V
    print(n,"cone_pixels:", cone_pixels.shape[0]) # rows

    if cone_pixels.shape[0] > 0:
        X_cone_list.append(cone_pixels) # if pixels array not NULL
    
    # background ROI drawing for 3 times
    for k in range(1,4):
        plt.figure()
        plt.imshow(img_rgb)
        plt.title(f"{index}/{len(names)} \n draw BG ROI double-click to finish")
        roi_bg = RoiPoly(color='b')
        mask_bg = roi_bg.get_mask(gray)
        plt.close()

        bg_pixels = img_hsv[mask_bg]
        print("   bg", k, "pixels:", bg_pixels.shape[0])

        if bg_pixels.shape[0] > 0:
            X_bg_list.append(bg_pixels)






# vertical stack（竖着拼）combined into big matrix data
X_cone = np.vstack(X_cone_list) if len(X_cone_list) else np.zeros((0, 3), dtype=np.uint8)
X_bg = np.vstack(X_bg_list) if len(X_bg_list) else np.zeros((0, 3), dtype=np.uint8)

np.save("X_cone_hsv.npy", X_cone)
np.save("X_bg_hsv.npy", X_bg)

print("saved X_cone_hsv.npy:", X_cone.shape)
print("saved X_bg_hsv.npy:", X_bg.shape)

