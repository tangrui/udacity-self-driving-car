import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas

# Constants
BASE_DIR = './data'
#BASE_DIR = '/Volumes/Backup/Archive/Training Data/Udacity/SDCND/data3'
COLUMN_NAMES = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
PATH_SEPERATOR = '/'
ANGLE_CORRECTION = 0.25
TRANS_X_RANGE = 100
TRANS_Y_RANGE = 40
ANGLE_PER_TRANS = 0.15
SHADOW_VALUE = 0.5
IMG_ROWS = 66
IMG_COLS = 200
IMG_CHS = 3

def read_samples():
    """
    Read driving_log.csv into pandas dataframe.
    """
    samples = pandas.read_csv('{}/driving_log.csv'.format(BASE_DIR), names=COLUMN_NAMES)
    return samples

def transform_image_path(original_image_path):
    """
    Transform the original absolute image path to relative path.
    """
    image_name = original_image_path.split(PATH_SEPERATOR)[-1].strip()
    return '{}/IMG/{}'.format(BASE_DIR, image_name)

def randomly_choose_camera(sample):
    """
    Randomly choose which camera image to use and adjust the steering angle accordingly.
    """
    rand = np.random.randint(3)
    if rand == 0:
        path = sample.center
        angle = sample.steering
    elif rand == 1:
        path = sample.left
        angle = sample.steering + ANGLE_CORRECTION
    else:
        path = sample.right
        angle = sample.steering - ANGLE_CORRECTION

    return transform_image_path(path), angle

def change_brightness(image):
    """
    Randomly change the brightness of the image.
    """
    rand = np.random.uniform(0.3, 1.2)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * rand
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def generate_x_trans():
    """
    Generate translation value in x direction.
    """
    return TRANS_X_RANGE * np.random.uniform() - TRANS_X_RANGE / 2

def translate_image(image, x_trans):
    """
    Translate the image in x and y direction.
    """
    y_trans = TRANS_Y_RANGE * np.random.uniform() - TRANS_Y_RANGE / 2
    M = np.array([[1, 0, x_trans], [0, 1, y_trans]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def translate_angle(angle, x_trans):
    """
    Adjust the steering angle according the x translation.
    """
    return angle + x_trans / TRANS_X_RANGE * 2 * ANGLE_PER_TRANS

def flip(image, angle):
    """
    Flip the image left to right, and change the angle to its opposite.
    """
    return np.fliplr(image), -angle

def fit_line(p1, p2):
    """
    Given two points, and caculate the slop and intercept.
    """
    x1, y1 = p1
    x2, y2 = p2
    slop = (y2 - y1) / (x2 - x1)
    intercept = y1 - (slop * x1)
    return slop, intercept

def add_random_shadow(image):
    """
    Randomly add horizontal or vertical shadows to the image.
    """
    h, w = image.shape[0], image.shape[1]
    shadowed_image = np.copy(image)
    shadow_value = SHADOW_VALUE

    if np.random.randint(2) == 0: # add vertical shadow
        [x1, x2] = np.random.choice(w, 2, replace=False)
        a, b = fit_line((x1, 0), (x2, h))
        add_to_left = np.random.randint(2)
        for y in range(h):
            x = int((y - b) / a)
            if add_to_left:
                shadowed_image[y, :x, :] = (shadowed_image[y, :x, :] * shadow_value).astype(np.int8)
            else:
                shadowed_image[y, x:, :] = (shadowed_image[y, x:, :] * shadow_value).astype(np.int8)
    else: # add horizontal shadow
        [y1, y2] = np.random.choice(h, 2, replace=False)
        a, b = fit_line((0, y1), (w, y2))
        add_to_above = np.random.randint(2)
        for x in range(w):
            y = int(a * x + b)
            if add_to_above:
                shadowed_image[:y, x, :] = (shadowed_image[:y, x, :] * shadow_value).astype(np.int8)
            else:
                shadowed_image[y:, x, :] = (shadowed_image[y:, x, :] * shadow_value).astype(np.int8)
    return shadowed_image

def preprocess(image):
    """
    Crop and resize the image.
    """
    cropped_image = image[50:140, :, :]
    # Pay attention to the input shape when calling cv2.resize method
    return cv2.resize(cropped_image, (IMG_COLS, IMG_ROWS), interpolation=cv2.INTER_AREA)

def augment(path, angle, threshold, bias):
    x_trans = generate_x_trans()
    angle = translate_angle(angle, x_trans)
    if abs(angle) < threshold - bias or abs(angle) > 1.0:
        return None, None

    image = mpimg.imread(path)
    image = change_brightness(image)
    image = add_random_shadow(image)
    image = translate_image(image, x_trans)
    if np.random.randint(2) == 1:
        image, angle = flip(image, angle)
    image = preprocess(image)
    return image, angle

def train_data_generator(samples, bias, batch_size):
    x_train = []
    y_train = []
    count = 0
    samples = samples.sample(frac=1).reset_index(drop=True)
    while True:
        rand = np.random.randint(len(samples))
        sample = samples.iloc[rand]
        path, angle = randomly_choose_camera(sample)

        threshold = np.random.uniform()
        image, angle = augment(path, angle, threshold, bias)

        if image is not None:
            x_train.append(image)
            y_train.append(angle)
            count += 1

        if count == batch_size:
            yield np.array(x_train), np.array(y_train)

            x_train = []
            y_train = []
            count = 0

def validation_data_generator(samples, batch_size):
    x_valid = []
    y_valid = []
    count = 0
    samples = samples.sample(frac=1).reset_index(drop=True)
    while True:
        rand = np.random.randint(len(samples))
        sample = samples.iloc[rand]
        path, angle = randomly_choose_camera(sample)
        image, angle = augment(path, angle, 1, 1)
        x_valid.append(image)
        y_valid.append(angle)
        count += 1

        if count == batch_size:
            yield np.array(x_valid), np.array(y_valid)

            x_valid = []
            y_valid = []
            count = 0
