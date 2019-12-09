import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def test1():
    datagen = ImageDataGenerator(
            #rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            #shear_range=0.2,
            zoom_range=0.2,
            #horizontal_flip=True,
            fill_mode='nearest')

    img = load_img('../data/img_data/all/Acutes_NGS2_plot_01_.png')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='../data/img_data/temp', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 5:
            break  # otherwise the generator would loop indefinitely

def test():
    import cv2
    img = cv2.imread('../data/img_data/all/Acutes_NGS2_plot_03_.png')
    crop_img = img[20:445, 30:410]  # Crop from x, y, w, h -> 100, 200, 300, 400
    print len(img[20:445])
    print len(img[35:410])
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)

test()