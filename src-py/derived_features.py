import constants as cs
import preprocessor as ps
import numpy as np

def load_data(data_file, label_file , test_size=0.33):
    from sklearn.model_selection import train_test_split
    X = open(data_file, "r").read().splitlines()
    y = open(label_file, "r").read().splitlines()

    return train_test_split(X, y, test_size=test_size, random_state=42)



def get_data_mapping(my_dict, data) :
    tr_data=list()
    for seq in data :
        l=[]
        words=seq.split(cs.SEPARATOR)
        for word in words:
            for i in range(0, len(word) - cs.SEQ_LEN, 1):
                seq_in = word[i:i + cs.SEQ_LEN]
                if my_dict.has_key(seq_in) :
                    val=my_dict.get(seq_in)
                else :
                    val=my_dict.get("UNK")
                l.append(val)
        tr_data.append(l)

    return tr_data

def get_mapping(data):
    my_mapping={'UNK':0}
    indx=1
    dup=0
    for seq in data :
        words=seq.split(cs.SEPARATOR)
        for word in words:
            for i in range(0, len(word) - cs.SEQ_LEN, 1):
                seq_in = word[i:i + cs.SEQ_LEN]

                if not my_mapping.has_key(seq_in) :
                    my_mapping[seq_in]=indx
                    indx +=1
                else:
                    dup+=1
    print "Found duplicates: ", dup
    return my_mapping



def get_transformed_data():
    global n_words
    # Prepare training and testing data
    x_train, x_test, y_train, y_test = load_data(cs.DATA_DIR+cs.TRAIN_DATA_FILE_NAME,
                                                 cs.DATA_DIR+cs.TRAIN_DATA_LABEL_FILE_NAME)

    # Process vocabulary and get a mapping dict from training data
    mapping_dict = get_mapping(x_train)

    #Transform data into id's instead of strings
    x_transform_train = get_data_mapping(mapping_dict,x_train)
    x_transform_test = get_data_mapping(mapping_dict, x_test)

    x_train = x_transform_train
    x_test = x_transform_test
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    n_words = len(mapping_dict.keys())
    print('Total words: %d' % n_words)
    return  x_train, x_test, y_train, y_test, mapping_dict

def persist_data(x_train, x_test, y_train, y_test, mapping_dict):
    import cPickle

    datapath=cs.DATA_DIR


    with  open(datapath + cs.TRAIN_DATA_SEQ_FILE_NAME, 'w+') as myfile:
        cPickle.dump(x_train, myfile)

    with  open(datapath + cs.TRAIN_DATA_SEQ_LABEL_FILE_NAME, 'w+') as myfile:
        cPickle.dump(y_train, myfile)

    with  open(datapath + cs.TEST_DATA_SEQ_FILE_NAME, 'w+') as myfile:
        cPickle.dump(x_test, myfile)

    with  open(datapath + cs.TEST_DATA_SEQ_LABEL_FILE_NAME, 'w+') as myfile:
        cPickle.dump(y_test, myfile)

    with  open(datapath + cs.DATA_SEQ_MAPPING_FILE_NAME, 'w+') as myfile:
        cPickle.dump(mapping_dict, myfile)


def load_derived_data():
    import cPickle

    datapath=cs.DATA_DIR


    with  open(datapath + cs.TRAIN_DATA_SEQ_FILE_NAME, 'r') as myfile:
        x_train=cPickle.load(myfile)

    with  open(datapath + cs.TRAIN_DATA_SEQ_LABEL_FILE_NAME, 'r') as myfile:
        y_train=cPickle.load(myfile)

    with  open(datapath + cs.TEST_DATA_SEQ_FILE_NAME, 'r') as myfile:
        x_test=cPickle.load(myfile)

    with  open(datapath + cs.TEST_DATA_SEQ_LABEL_FILE_NAME, 'r') as myfile:
        y_test=cPickle.load(myfile)

    with  open(datapath + cs.DATA_SEQ_MAPPING_FILE_NAME, 'r') as myfile:
        mapping_dict=cPickle.load(myfile)

    return np.array(x_train), np.array(x_test), np.asarray(y_train, dtype=int), \
           np.asarray(y_test, dtype=int), mapping_dict


if __name__ == "__main__":
    __author__ = 'Sunitha Basodi'
    #ps.preprocess_data(cs.DATA_DIR+cs.RAW_DATA_DIR_NAME, cs.DATA_DIR)
    x_train, x_test, y_train, y_test, mapping_dict=get_transformed_data()
    persist_data(x_train, x_test, y_train, y_test, mapping_dict)


