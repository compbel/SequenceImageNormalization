import numpy as np
import pandas as pd
import image_data_utils as idu
import constants as cs
import image_data_classifiers as idc

#This file is to train LinearSVM model based on all Clinic/chronic data and predict labels for new unseen data
def build_model(type="lsvm"):
    try:
        model=idu.load_persisted_model(type)
    except:
        x_train, x_val, y_train, y_val = idc.get_data(crop_image=False, image_res_dir="img_480_480")
        x_all = np.concatenate((x_train, x_val), axis=0)
        y_all = np.concatenate((y_train, y_val), axis=0)
        model=idc.get_model(type, params={})
        model.fit(x_all, y_all)
        idu.persist_model(model, type)

    return model

def run_model(type, test_file_names, test_data, export_file_path):
    model=build_model(type=type)
    y_val_predict = model.predict(test_data)
    list_of_tuples = list(zip(test_file_names, y_val_predict, y_val_predict))
    df = pd.DataFrame(list_of_tuples, columns = ['FileName', 'PredictedLabel', 'PredictedLabelID'])
    df.loc[df['PredictedLabel'] == cs.CHRONIC_LABEL, 'PredictedLabel'] = cs.CHRONIC_LABEL_STRING
    df.loc[df['PredictedLabel'] == cs.CLINIC_LABEL, 'PredictedLabel'] = cs.CLINIC_LABEL_STRING

    df.to_csv (export_file_path, index = None, header=True)

    return

def test():
    type="lsvm"
    #type="dt"
    res_dir="../results/"+type+"_CN_predicted_results.csv"
    test_data, test_file_names= idu.get_image_array_data_with_filenames("../data/CN/img_data/img_480_480/all/", merge_data_cols=True)
    run_model(type, test_file_names, test_data, res_dir)
    print("Stored result in: " + res_dir)

    return


if __name__ == '__main__':
    test()












