import os
import numpy as np # linear algebra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.util import montage as montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable() # memory is tight
from keras import models
from keras.preprocessing.image import load_img


MODEL_IMG_SIZE = (768, 768)
TEST_PERCENTAGE = 0.3
SAMPLES_PER_GROUP = 6000
BATCH_SIZE = 48
# downsampling in preprocessing
IMG_SCALING = (3, 3)

# number of validation images to uset_x
VALID_IMG_COUNT = 1000

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = '../../'
train_image_dir = os.path.join(ship_dir, 'train_v2')
from keras.optimizers import Adam
import keras.backend as K
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

def np_per_image_accuracy(y_true, y_pred):
    label_fn = lambda a: label(K.eval(a), return_num=True)[1]
    num_true = K.map_fn(label_fn, y_true)
    num_pred = K.map_fn(label_fn, y_pred)

    return K.mean(K.equal(num_true, num_pred), axis=-1)

fullres_model = models.load_model("model_fullres_keras.h5", custom_objects={'IoU': IoU})
# fullres_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])
# test_image_dir = os.path.join(ship_dir, 'test')

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))
not_empty = pd.notna(masks.EncodedPixels)
print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')
print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
# filter unavailable files
# unique_img_ids['is_available'] = unique_img_ids['ImageId'].map(
#     lambda imgid: os.path.isfile(os.path.join(train_image_dir, imgid)))
# unique_img_ids = unique_img_ids[unique_img_ids['is_available']]

unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
# some files are too small/corrupt
# unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
#                                                                os.stat(os.path.join(train_image_dir,
#                                                                                     c_img_id)).st_size/1024)
# unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 20] # keep only +50kb files
# unique_img_ids['file_size_kb'].hist()
masks.drop(['ships'], axis=1, inplace=True)

# SAMPLES_PER_GROUP = (unique_img_ids[unique_img_ids['ships']==1]['ImageId'].count() + unique_img_ids[unique_img_ids['ships']==2]['ImageId'].count())//3
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)
balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1).figure.savefig("samples-dist.png")

from sklearn.model_selection import train_test_split
train_ids, valid_ids = train_test_split(balanced_train_df,
                 test_size = TEST_PERCENTAGE,
                 #stratify = balanced_train_df['ships']
                )
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

def make_image_gen(in_df, batch_size = BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

def raw_prediction(img, path):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    return cur_seg, c_img[0]

def smooth(cur_seg):
    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))

def predict(img, path):
    cur_seg, c_img = raw_prediction(img, path=path)
    return smooth(cur_seg), c_img

# Basic validation
num_true_list = []
num_pred_list = []
max_true_list = []
max_pred_list = []
binary_accuracy_list = []

all_batches = list(valid_df.groupby('ImageId'))
np.random.shuffle(all_batches)
for c_img_id, c_masks in all_batches:
    img = load_img(os.path.join(train_image_dir, c_img_id), target_size=MODEL_IMG_SIZE)
    img = np.expand_dims(img, 0)/255.0
    valid_y = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
    pred_y = fullres_model.predict(img)[0]

    label_fn = lambda a: label(a, return_num=True)[1]

    num_true_list.append(label_fn(valid_y))
    num_pred_list.append(label_fn(pred_y))
    max_true_list.append(valid_y.max())
    max_pred_list.append(pred_y.max())
    binary_accuracy_list.append((valid_y == pred_y).astype('uint8').mean())

num_true_list = np.array(num_true_list)
num_pred_list = np.array(num_pred_list)

max_true_list = np.array(max_true_list)
max_pred_list = np.array(max_pred_list)

binary_accuracy_list = np.array(binary_accuracy_list)

print(num_true_list.shape, (num_true_list == num_pred_list).astype('uint8').mean())
print("Zero ship images count: true {}, pred {}".format((max_true_list == 0).size, (max_pred_list == 0).size)
print("Binary Accuracy: "+binary_accuracy_list.mean())

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.hist((num_true_list == num_pred_list), np.linspace(0, 1, 20))
ax.set_xlim(0, 1)
ax.set_yscale('log', nonposy='clip')
fig.savefig("validate.png")
print("validate.png saved")

## Get a sample of each group of ship count
samples = valid_df.groupby('ships').apply(lambda x: x.sample(1))
fig, m_axs = plt.subplots(samples.shape[0], 4, figsize = (15, samples.shape[0]*4))
[c_ax.axis('off') for c_ax in m_axs.flatten()]

for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):
    first_seg, first_img = raw_prediction(c_img_name, train_image_dir)
    ax1.imshow(first_img)
    ax1.set_title('Image: ' + c_img_name)
    ax2.imshow(first_seg[:, :, 0], cmap=get_cmap('jet'))
    ax2.set_title('Model Prediction')
    reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))
    ax3.imshow(reencoded)
    ax3.set_title('Prediction Masks')
    ground_truth = masks_as_color(masks.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels'])
    ax4.imshow(ground_truth)
    ax4.set_title('Ground Truth')

fig.savefig('validation.png')