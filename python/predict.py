from skimage.io import imread
import numpy as np
from keras import models
from skimage.morphology import binary_opening, disk

fullres_model = models.load_model("model_fullres_keras.h5")

def _raw_prediction(img):
    img = np.expand_dims(img, 0)/255.0
    seg = fullres_model.predict(img)[0]
    return seg, img[0]

def smooth(seg):
    return binary_opening(seg>0.99, np.expand_dims(disk(2), -1))

def predict_by_path(img_path):
    img = imread(img_path)
    seg, img = _raw_prediction(img)
    # return smooth(cur_seg), c_img
    return seg, img

# seg, img = predict_by_path("")

# ax2.imshow(first_seg[:, :, 0], cmap=get_cmap('jet'))
# reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))
