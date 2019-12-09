
from sklearn.model_selection import train_test_split
import glob
import constants as cs
import numpy as np
from matplotlib.image import imread

def persist_img_numpy_data(crop_image=False, test_size=0.2):
    import gzip, cPickle, cv

    chr_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CHRONIC_DIR_NAME + "*")
    clinic_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CLINIC_DIR_NAME + "*")
    master_dataset=[]
    master_labels=[]
    for dir in [chr_files, clinic_files] :
        label = cs.CHRONIC_LABEL if cs.CHRONIC_DIR_NAME in dir[0] else cs.CLINIC_LABEL
        for filename in dir:
            x = cv.LoadImageM(filename)
            data = np.asarray(x)

            #print data.shape

            master_dataset.append(data)
            master_labels.append(label)

    print (len(master_labels), len(master_dataset))
    X_train, X_test, y_train, y_test = train_test_split(master_dataset, master_labels,
                                                            test_size=test_size, random_state=42)
    train_set = np.array(X_train), np.asarray(y_train)

    test_set = np.array(X_test), np.asarray(y_test)

    dataset = [train_set, test_set]

    print("Creating pickle file")
    out_filename=cs.NUMPY_CROPPED_IMG_FILE if crop_image else cs.NUMPY_IMG_FILE;

    f = gzip.open(cs.IMAGE_DATA_DIR +out_filename, 'wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()


def load_img_numpy_data(crop_image=False):
    import cPickle, gzip
    out_filename=cs.NUMPY_CROPPED_IMG_FILE if crop_image else cs.NUMPY_IMG_FILE;

    f = gzip.open(cs.IMAGE_DATA_DIR +out_filename)
    data = cPickle.load(f)

    x_train, y_train= data[0]
    x_test, y_test= data[1]
    return  x_train, x_test, y_train, y_test

def get_image_array_labelled_data_cv2():
    import cv2.cv as cv,pandas as pd, os
    #import cv2, pandas as pd
    #chr_df = pd.read_csv(cs.IMAGE_DATA_DIR+cs.CHRONIC_NUM_SEQ, header=None)
    #clinic_df = pd.read_csv(cs.IMAGE_DATA_DIR+cs.CLINIC_NUM_SEQ, header=None)

    num_rec_list=[]

    chr_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CHRONIC_DIR_NAME + "*")
    clinic_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CLINIC_DIR_NAME + "*")

    master_dataset=[]
    master_labels=[]
    for dir in [chr_files, clinic_files] :
        label = cs.CHRONIC_LABEL if cs.CHRONIC_DIR_NAME in dir[0] else cs.CLINIC_LABEL
        for filename in dir:
            #print filename, dir
            #num_rec_list.append( os.path.basename(filename).split('.')[1].split('_')[1])
            num_rec_list=100#.append( os.path.basename(filename).split('.')[1].split('_')[1])
            x = cv.LoadImageM(filename)
            data = np.asarray(x)

            #x = cv2.imread(filename)
            #data = np.asarray(x)
            #print (data.shape)

            master_dataset.append(data)
            master_labels.append(label)

    print( len(master_labels), len(master_dataset))

    x_data = np.array(master_dataset)
    y_data = np.asarray(master_labels)
    num_rec = np.array(num_rec_list)
    #num_rec.shape= (len(num_rec_list), 1)
    
 
    print "Loading Clinic,Chronic files: ",cs.IMAGE_DATA_DIR_ALL,"\nX shape: ", x_data.shape, "\t y_data shape: ", y_data.shape, "\t num_rec shape: ", num_rec.shape

    return  x_data, y_data, num_rec


def get_image_array_data(crop_image): 
    from matplotlib.image import imread
    import os

    dir_name=cs.OUTBREAKS_IMG_DIR_ALL
    if crop_image :
        dir_name=cs.OUTBREAKS_IMG_DIR_ALL_CROP

    print "Reading outbreak files from: ", dir_name
    files = glob.glob(dir_name + "*" )
    #print files[0]

    master_dataset = []
    master_labels = []
    master_filenames = []
    for filename in sorted(files):
        # print filename
        master_dataset.append(imread(filename))
        file_basename=os.path.basename(filename)
        master_filenames.append(file_basename)
        master_labels.append(file_basename.split("_")[1][0:2])

    #print("Length of files read: ", len(master_filenames), len(master_dataset))

    return  np.array(master_dataset), master_labels, master_filenames


def get_consensus_array_data(withXX):
    from matplotlib.image import imread
    import os

    dir_name=cs.OUTBREAKS_CONSENSES_DIR_WITH_XX if withXX else cs.OUTBREAKS_CONSENSES_DIR_NO_XX

    print "Reading outbreak CONSENSES files from: ", dir_name
    files = glob.glob(dir_name + "*.txt" )
    #print files[0]

    master_dataset = []
    master_labels = []
    master_filenames = []
    for filename in sorted(files):
        # print filename
        with open(filename, 'r') as file:
            data = file.read().replace('\n', '')
        master_dataset.append(data)
        file_basename=os.path.basename(filename)
        master_filenames.append(file_basename)
        master_labels.append(file_basename.split("_")[1][0:2])

    #print("Length of files read: ", len(master_filenames), len(master_dataset))

    return  np.array(master_dataset), master_labels, master_filenames


def get_outbreak_actual_clusters():
     import ast
     return  ast.literal_eval(cs.OUTBREAKS_ACTUAL_CLUSTERS)


def get_image_array_labelled_data():
    from matplotlib.image import imread

    num_rec_list = []

    chr_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CHRONIC_DIR_NAME + "*")
    clinic_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CLINIC_DIR_NAME + "*")

    master_dataset = []
    master_labels = []
    for dir in [chr_files, clinic_files]:
        #print("dir: ", dir)
        label = cs.CHRONIC_LABEL if cs.CHRONIC_DIR_NAME in dir[0] else cs.CLINIC_LABEL
        for filename in dir:
            # print filename, dir
            #num_rec_list.append( os.path.basename(filename).split('.')[1].split('_')[1])
            num_rec_list = [100]  #TODO .append( os.path.basename(filename).split('.')[1].split('_')[1])
            data = imread(filename)
            #print (data.shape)

            # x = cv2.imread(filename)
            # data = np.asarray(x)
            # print (data.shape)

            master_dataset.append(data)
            master_labels.append(label)

    print(len(master_labels), len(master_dataset))

    x_data = np.array(master_dataset)
    y_data = np.asarray(master_labels)
    num_rec = np.array(num_rec_list)
    num_rec.shape = (len(num_rec_list), 1)

    print "Loading Clinic,Chronic files: ",cs.IMAGE_DATA_DIR_ALL,"\nX shape: ", x_data.shape, "\t y_data shape: ", y_data.shape, "\t num_rec shape: ", num_rec.shape

    return x_data, y_data, num_rec

def reshape_pixel_data(Xtr):
    return Xtr.reshape(Xtr.shape[0], Xtr.shape[1] * Xtr.shape[2] * Xtr.shape[3])


def get_image_array_data_with_filenames(dir_name, merge_data_cols=True):
    file_names = []
    master_dataset = []

    all_img_files = glob.glob(dir_name + "*.png")

    for filename in all_img_files:
        # print filename, dir
        data = imread(filename)
        #print (data.shape)

        master_dataset.append(data)
        file_names.append(filename)

    print(len(file_names), len(master_dataset))

    x_data = np.array(master_dataset)

    print "dir_name: ",dir_name,"\nX shape: ", x_data.shape, "\t len of file names : ", len(file_names)
    if merge_data_cols: 
        x_data=reshape_pixel_data(x_data)

    return x_data,file_names


def get_random_indices(total_items, sample_size_needed):
    import random
    return random.sample(range(total_items), sample_size_needed)


def get_image_array_balanced_data(test_size=0.2):

    from matplotlib.image import imread


    num_rec_list = []

    chr_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CHRONIC_DIR_NAME + "*")
    clinic_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CLINIC_DIR_NAME + "*")

    data_size=min(len(chr_files), len(clinic_files))

    valid_chr_indices=get_random_indices(max(len(chr_files), len(clinic_files)), min(len(chr_files), len(clinic_files)))

    chr_sample_files = [chr_files[i] for i in valid_chr_indices]

    print "Length of chr sample: ", len(chr_sample_files)
    print "Length of Acute sample: ", len(clinic_files) 

    master_dataset = []
    master_labels = []
    for dir in [chr_sample_files, clinic_files]:
        label = cs.CHRONIC_LABEL if cs.CHRONIC_DIR_NAME in dir[0] else cs.CLINIC_LABEL
        for filename in dir:
            # print filename, dir
            #num_rec_list.append( os.path.basename(filename).split('.')[1].split('_')[1])
            num_rec_list = [100]  #TODO .append( os.path.basename(filename).split('.')[1].split('_')[1])
            data = imread(filename)
            #print (data.shape)

            # x = cv2.imread(filename)
            # data = np.asarray(x)
            # print (data.shape)

            master_dataset.append(data)
            master_labels.append(label)

    print(len(master_labels), len(master_dataset))

    x_data = np.array(master_dataset)
    y_data = np.asarray(master_labels)
    num_rec = np.array(num_rec_list)
    num_rec.shape = (len(num_rec_list), 1)

    print "X shape: ", x_data.shape, "\t y_data shape: ", y_data.shape, "\t num_rec shape: ", num_rec.shape

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    x_train = np.array(x_train)
    y_train = np.asarray(y_train)
    x_test = np.array(x_test)
    y_test = np.asarray(y_test)

    return x_train, x_test, y_train, y_test


def get_image_array_split_data(test_size=0.2):
    x_data, y_data, num_rec = get_image_array_labelled_data()
    num_rec_train =[]
    num_rec_test=[]
    #TODO uncomment
    # x_train, x_test, y_train, y_test, num_rec_train, num_rec_test = train_test_split(x_data, y_data, num_rec, test_size=test_size, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    x_train = np.array(x_train)
    y_train = np.asarray(y_train)
    x_test = np.array(x_test)
    y_test = np.asarray(y_test)

    return x_train, x_test, y_train, y_test, num_rec_train, num_rec_test

def persist_model(model, model_name):
    import os
    from joblib import dump
    if not os.path.exists(cs.MODEL_PERSIST_PATH): os.makedirs(cs.MODEL_PERSIST_PATH)
    dump(model, cs.MODEL_PERSIST_PATH+model_name+'.joblib')

def load_persisted_model(model_name) :
    from joblib import load
    return load(cs.MODEL_PERSIST_PATH+model_name+'.joblib')
