# %%
import numpy as np

import matplotlib.pyplot as plt

from skimage.measure import label
from skimage import data
from skimage import color
from skimage.morphology import extrema
from skimage import exposure
from PIL import Image
from skimage.feature import peak_local_max


# %%
img = Image.open('C:\\Users\\Dell\\Desktop\\od_zacatku\\train_img\\train4.tiff')
imarray = np.array(img)
h = 0.3
x_0 = 70
y_0 = 354
width = 256
height = 256

imarray = exposure.rescale_intensity(imarray)

local_maxima = extrema.local_maxima(imarray)
label_maxima = label(local_maxima)
print(label_maxima.shape)
print(imarray.shape)
print(label_maxima.dtype)
print(imarray.dtype)

overlay = color.label2rgb(label_maxima, imarray, alpha=0.3, bg_label=0,
                          bg_color=None, colors=[(1, 0, 0)])


h_maxima = extrema.h_maxima(imarray, h)
label_h_maxima = label(h_maxima)
overlay_h = color.label2rgb(label_h_maxima, imarray, alpha=0.3, bg_label=0,
                            bg_color=None, colors=[(1, 0, 0)])





# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(imarray[y_0:(y_0 + height), x_0:(x_0 + width)], cmap='gray',
             interpolation='none')
ax[0].set_title('Original image')
ax[0].axis('off')

ax[1].imshow(overlay[y_0:(y_0 + height), x_0:(x_0 + width)],
             interpolation='none')
ax[1].set_title('Local Maxima')
ax[1].axis('off')

ax[2].imshow(overlay_h[y_0:(y_0 + height), x_0:(x_0 + width)],
             interpolation='none')
ax[2].set_title('h maxima for h = %.2f' % h)
ax[2].axis('off')
plt.show()

# %%
