import numpy as np
from keras import models
from keras.preprocessing.image import load_img
from PIL import Image
from skimage.morphology import binary_opening, disk, label
from skimage.measure import regionprops
import os

fullres_model = None
MODEL_IMG_SIZE = (768, 768)

def load_model():
    global fullres_model
    fullres_model = models.load_model("model_fullres_keras.h5")
    if os.environ.get('SHIPDETECTION_BROKEN_MODEL'):
        from keras.optimizers import Adam
        import keras.backend as K
        def IoU(y_true, y_pred, eps=1e-6):
            if np.max(y_true) == 0.0:
                return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
            intersection = K.sum(y_true * y_pred, axis=[1,2,3])
            union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
            return -K.mean( (intersection + eps) / (union + eps), axis=0)
        fullres_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])

def _raw_prediction(img):
    global fullres_model
    if not fullres_model:
        load_model()

    img = np.expand_dims(img, 0)/255.0
    seg = fullres_model.predict(img)[0]
    return seg, img[0]

def smooth(seg):
    return binary_opening(seg>0.99, np.expand_dims(disk(2), -1)[:,:,0]).astype('uint8')

def predict_by_path(img_path):
    img = load_img(img_path, target_size=MODEL_IMG_SIZE)
    # load_img take care of RGBA images by itself;
    # it might have transparency issue like https://stackoverflow.com/q/9166400/4073795
    seg, img = _raw_prediction(img)
    seg = seg[:, :, 0]
    return smooth(seg), img
    # return seg, img

def save_by_path(seg, path):
    seg = seg.astype('uint8')*255
    return Image.fromarray(seg).convert("RGB").save(path)

def extract_seg(seg):
    labels = label(seg)
    regions_original = regionprops(labels)
    regions = [region.bbox for region in regions_original if region.area > 100]
    # bbox: (min_row, min_col, max_row, max_col)

    return {'regions': regions, 'size': MODEL_IMG_SIZE}

# seg, img = predict_by_path("")
