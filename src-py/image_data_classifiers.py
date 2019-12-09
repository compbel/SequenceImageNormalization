import numpy as np
import image_data_utils as idu
import constants as cs


class NearestNeighbor(object):
    def __init__(self, k):
        self.k = k
        pass

    def fit(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            # using the L1 distance (sum of absolute value differences)
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))

            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred


def custom_nn_test(x_train, x_val, y_train, y_val, type, params):
    import gc
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics
    from sklearn.metrics import precision_recall_fscore_support

    # find hyperparameters that work best on the validation set

    acc, prec_0, recall_0, f1_score_0, support_0 = 0, 0, 0, 0, 0
    prec_1, recall_1, f1_score_1, support_1, auc = 0, 0, 0, 0, 0
    cv_FLAG = params['cv']

    if cv_FLAG:
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)
        x_all = np.concatenate((x_train, x_val), axis=0)
        y_all = np.concatenate((y_train, y_val), axis=0)

        # Numpy arrays to store the result of each iteration; metric_0 => CLINIC_LABEL(-1) and  metric_1 => CHRONIC_LABEL(+1)
        acc_arr, prec_0_arr, recall_0_arr, f1_score_0_arr, support_0_arr = (np.zeros(n_splits) for i in range(5))
        prec_1_arr, recall_1_arr, f1_score_1_arr, support_1_arr, auc_arr = (np.zeros(n_splits) for i in range(5))

        counter = 0
        for train_index, test_index in skf.split(x_all, y_all):
            cvx_train, cvx_test = x_all[train_index], x_all[test_index]
            cvy_train, cvy_test = y_all[train_index], y_all[test_index]
            # use a particular value of k and evaluation on validation data
            model = get_model(type, params)

            model.fit(cvx_train, cvy_train)
            # here we assume a modified NearestNeighbor class that can take a k as input
            Yval_predict = model.predict(cvx_test)
            acc1 = np.mean(Yval_predict == cvy_test)

            fpr, tpr, thresholds = metrics.roc_curve(cvy_test, Yval_predict)
            auc1 = metrics.auc(fpr, tpr)

            p, r, f, s = precision_recall_fscore_support(cvy_test, Yval_predict)

            # Update Numpy arrays
            acc_arr[counter] = acc1
            prec_0_arr[counter] = p[0]
            recall_0_arr[counter] = r[0]
            f1_score_0_arr[counter] = f[0]
            support_0_arr[counter] = s[0]
            prec_1_arr[counter] = p[1]
            recall_1_arr[counter] = r[1]
            f1_score_1_arr[counter] = f[1]
            support_1_arr[counter] = s[1]
            auc_arr[counter] = auc1
            counter += 1

            acc += acc1
            prec_0 += p[0]
            recall_0 += r[0]
            f1_score_0 += f[0]
            support_0 += s[0]
            prec_1 += p[1]
            recall_1 += r[1]
            f1_score_1 += f[1]
            support_1 += s[1]
            auc += auc1
            gc.collect()
        acc = acc / n_splits
        prec_0 = prec_0 / n_splits
        recall_0 = recall_0 / n_splits
        f1_score_0 = f1_score_0 / n_splits
        support_0 = support_0 / n_splits

        prec_1 = prec_1 / n_splits
        recall_1 = recall_1 / n_splits
        f1_score_1 = f1_score_1 / n_splits
        support_1 = support_1 / n_splits
        auc = auc / n_splits

        # Persist the arrays : Concatenate all numpy arrays and save to file
        # acc_arr,  prec_0_arr,  recall_0_arr,  f1_score_0_arr,  support_0_arr,  prec_1_arr,  recall_1_arr,  f1_score_1_arr,  support_1_arr,  auc_arr
        # print acc_arr
        npfinal = np.concatenate(([acc_arr], [prec_0_arr], [recall_0_arr], [f1_score_0_arr], [support_0_arr],
                                  [prec_1_arr], [recall_1_arr], [f1_score_1_arr], [support_1_arr], [auc_arr]), axis=0)
        # print npfinal

        my_header = cs.IMAGE_DATA_DIR + "-type-" + type + "-CV-" + str(cv_FLAG) + "-params-" + str(params);
        np.savetxt(my_header + ".csv", npfinal, delimiter=',', header=my_header,
                   comments="__acc_arr::prec_0_arr::recall_0_arr::f1_score_0_arr::support_0_arr::prec_1_arr::recall_1_arr::f1_score_1_arr::support_1_arr::auc_arr_____")

    else:
        model = get_model(type, params)
        model.fit(x_train, y_train)
        # here we assume a modified NearestNeighbor class that can take a k as input
        Yval_predict = model.predict(x_val)
        acc = np.mean(Yval_predict == y_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, Yval_predict)
        auc += metrics.auc(fpr, tpr)
        p, r, f, s = precision_recall_fscore_support(y_val, Yval_predict)
        prec_0 += p[0]
        recall_0 += r[0]
        f1_score_0 += f[0]
        support_0 += s[0]
        prec_1 += p[1]
        recall_1 += r[1]
        f1_score_1 += f[1]
        support_1 += s[1]

        npfinal = np.concatenate(([acc], [prec_0], [recall_0], [f1_score_0], [support_0], [prec_1], [recall_1],
                                  [f1_score_1], [support_1], [auc]), axis=0)

        my_header = cs.IMAGE_DATA_DIR + cs.IMAGE_RESOLUTION_DIR[:-1] + "-type-" + type + "-randomSampling-" + str(
            cv_FLAG) + "-params-" + str(params);
        print my_header
        np.savetxt(my_header + ".csv", npfinal, delimiter=',', header=my_header,
                   comments="__acc_arr::prec_0_arr::recall_0_arr::f1_score_0_arr::support_0_arr::prec_1_arr::" +
                            "recall_1_arr::f1_score_1_arr::support_1_arr::auc_arr_____")

    t = [cs.IMAGE_RESOLUTION_DIR, type, cv_FLAG, acc, prec_0, prec_1, recall_0, recall_1, f1_score_0, f1_score_1,
         support_0, support_1, auc, params]
    print ("\t".join([str(x) for x in t]))


def custom_nn_roc(x_train, x_val, y_train, y_val, type, params):
    import gc
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics
    from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
    from scipy import interp
    import matplotlib.pyplot as plt
    import pandas as pd

    # find hyperparameters that work best on the validation set

    acc, prec_0, recall_0, f1_score_0, support_0 = 0, 0, 0, 0, 0
    prec_1, recall_1, f1_score_1, support_1, auc = 0, 0, 0, 0, 0
    cv_FLAG = params['cv']
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if cv_FLAG:
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)
        x_all = np.concatenate((x_train, x_val), axis=0)
        y_all = np.concatenate((y_train, y_val), axis=0)
        data=[]

        # Numpy arrays to store the result of each iteration; metric_0 => CLINIC_LABEL(-1) and  metric_1 => CHRONIC_LABEL(+1)
        acc_arr, prec_0_arr, recall_0_arr, f1_score_0_arr, support_0_arr = (np.zeros(n_splits) for i in range(5))
        prec_1_arr, recall_1_arr, f1_score_1_arr, support_1_arr, auc_arr = (np.zeros(n_splits) for i in range(5))

        counter = 0
        for train_index, test_index in skf.split(x_all, y_all):
            cvx_train, cvx_test = x_all[train_index], x_all[test_index]
            cvy_train, cvy_test = y_all[train_index], y_all[test_index]
            # use a particular value of k and evaluation on validation data
            model = get_model(type, params)

            model.fit(cvx_train, cvy_train)
            probas_ = model.fit(cvx_train, cvy_train).predict_proba(cvx_test)

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(cvy_test, probas_[:, 1])
            print "Actual: ", fpr, tpr
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # aucs.append(roc_auc)

            # here we assume a modified NearestNeighbor class that can take a k as input
            Yval_predict = model.predict(cvx_test)
            acc1 = np.mean(Yval_predict == cvy_test)

            tempcv1=np.stack(( probas_[:, 1], Yval_predict, cvy_test), axis=-1)
            np.savetxt("../cv-"+str(counter) + ".csv", tempcv1, delimiter=',', header='crossValidationRes_predictProb_yPred_yTrue',
                   comments="crossValidationRes_predictProb_yPred_yTrue")

            fpr, tpr, thresholds = metrics.roc_curve(cvy_test, Yval_predict)
            print "Formula: ", fpr, tpr
            auc1 = metrics.auc(fpr, tpr)
            aucs.append(auc1)

            p, r, f, s = precision_recall_fscore_support(cvy_test, Yval_predict)

            # Update Numpy arrays
            acc_arr[counter] = acc1
            prec_0_arr[counter] = p[0]
            recall_0_arr[counter] = r[0]
            f1_score_0_arr[counter] = f[0]
            support_0_arr[counter] = s[0]
            prec_1_arr[counter] = p[1]
            recall_1_arr[counter] = r[1]
            f1_score_1_arr[counter] = f[1]
            support_1_arr[counter] = s[1]
            auc_arr[counter] = auc1
            counter += 1

            acc += acc1
            prec_0 += p[0]
            recall_0 += r[0]
            f1_score_0 += f[0]
            support_0 += s[0]
            prec_1 += p[1]
            recall_1 += r[1]
            f1_score_1 += f[1]
            support_1 += s[1]
            auc += auc1
            gc.collect()

        print "tpr, aucs  "
        print tpr
        print aucs

        acc = acc / n_splits
        prec_0 = prec_0 / n_splits
        recall_0 = recall_0 / n_splits
        f1_score_0 = f1_score_0 / n_splits
        support_0 = support_0 / n_splits

        prec_1 = prec_1 / n_splits
        recall_1 = recall_1 / n_splits
        f1_score_1 = f1_score_1 / n_splits
        support_1 = support_1 / n_splits
        auc = auc / n_splits

        # Persist the arrays : Concatenate all numpy arrays and save to file
        # acc_arr,  prec_0_arr,  recall_0_arr,  f1_score_0_arr,  support_0_arr,  prec_1_arr,  recall_1_arr,  f1_score_1_arr,  support_1_arr,  auc_arr
        # print acc_arr
        npfinal = np.concatenate(([acc_arr], [prec_0_arr], [recall_0_arr], [f1_score_0_arr], [support_0_arr],
                                  [prec_1_arr], [recall_1_arr], [f1_score_1_arr], [support_1_arr], [auc_arr]), axis=0)
        # print npfinal

        my_header = cs.IMAGE_DATA_DIR + "-type-" + type + "-CV-" + str(cv_FLAG) + "-params-" + str(params);
        np.savetxt(my_header + ".csv", npfinal, delimiter=',', header=my_header,
                   comments="__acc_arr::prec_0_arr::recall_0_arr::f1_score_0_arr::support_0_arr::prec_1_arr::recall_1_arr::f1_score_1_arr::support_1_arr::auc_arr_____")
        '''
        #PLOT
        #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            # label='Chance', alpha=.8)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        # save the final plot
    
        plt.savefig("../roc_cv.png", dpi=600)
        plt.savefig("../roc_cv.pdf", dpi=300)
        '''

    else:
        model = get_model(type, params)
        model.fit(x_train, y_train)
        # here we assume a modified NearestNeighbor class that can take a k as input
        Yval_predict = model.predict(x_val)
        acc = np.mean(Yval_predict == y_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, Yval_predict)
        auc += metrics.auc(fpr, tpr)
        p, r, f, s = precision_recall_fscore_support(y_val, Yval_predict)
        prec_0 += p[0]
        recall_0 += r[0]
        f1_score_0 += f[0]
        support_0 += s[0]
        prec_1 += p[1]
        recall_1 += r[1]
        f1_score_1 += f[1]
        support_1 += s[1]

        npfinal = np.concatenate(([acc], [prec_0], [recall_0], [f1_score_0], [support_0], [prec_1], [recall_1],
                                  [f1_score_1], [support_1], [auc]), axis=0)

        my_header = cs.IMAGE_DATA_DIR + cs.IMAGE_RESOLUTION_DIR[:-1] + "-type-" + type + "-randomSampling-" + str(
            cv_FLAG) + "-params-" + str(params);
        print my_header
        np.savetxt(my_header + ".csv", npfinal, delimiter=',', header=my_header,
                   comments="__acc_arr::prec_0_arr::recall_0_arr::f1_score_0_arr::support_0_arr::prec_1_arr::" +
                            "recall_1_arr::f1_score_1_arr::support_1_arr::auc_arr_____")

    t = [cs.IMAGE_RESOLUTION_DIR, type, cv_FLAG, acc, prec_0, prec_1, recall_0, recall_1, f1_score_0, f1_score_1,
         support_0, support_1, auc, params]
    print ("\t".join([str(x) for x in t]))


def get_decision_tree():
    from sklearn import tree
    return tree.DecisionTreeClassifier()


def scikit_knn(k, distance='minkowski'):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=k, metric=distance)


def scikit_nn(k, distance='minkowski'):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k, metric=distance, algorithm='ball_tree')


def get_lsvm(params):
    from sklearn.svm import LinearSVC
    return LinearSVC(random_state=0)


def get_random_forest(params):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=2, random_state=0)


def get_guassian_nb(params):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB();


def get_SGD(params):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(loss="hinge", penalty="l2")


def get_svc_linear(params):
    from sklearn.svm import SVC
    return SVC(kernel='linear', probability=True)


def get_svc_rbf(params):
    from sklearn.svm import SVC
    return SVC(kernel='rbf')


def get_model(type, params):
    if type == "scikit_knn":
        model = scikit_knn(params['k'], params['distance'])
    elif type == "scikit_nn":
        model = scikit_nn(params['k'], params['distance'])
    elif type == "dt":
        model = get_decision_tree()
    elif type == "lsvm":
        model = get_lsvm(params)
    elif type == "random_forest":
        model = get_random_forest(params)
    elif type == "guassian_nb":
        model = get_guassian_nb(params)
    elif type == "SGD":
        model = get_SGD(params)
    elif type == "svc_linear":
        model = get_svc_linear(params)
    elif type == "svc_rbf":
        model = get_svc_rbf(params)
    else:
        model = NearestNeighbor(params['k'])

    return model




def get_data(crop_image, image_res_dir, test_size=0.2):
    # if split == True:
    # Xtr, Xte, y_train, y_val=idu.load_img_numpy_data(crop_image=crop_image)
    cs.updateFileNames(image_res_dir, crop_image=False, sort=False)
    print " cs.IMAGE_RESOLUTION_DIR : " + cs.IMAGE_RESOLUTION_DIR[:-1]
    Xtr, Xte, y_train, y_val, num_rec_train, num_rec_test = idu.get_image_array_split_data(test_size=test_size)
    print(Xtr.shape, Xte.shape)
    # x_train = Xtr  # Xtr_rows becomes 50000 x 3072
    # x_val = Xte  # Xte_rows becomes 10000 x 3072

    assert Xtr.shape[1] == Xte.shape[1] and Xtr.shape[2] == Xte.shape[2] and Xtr.shape[3] == Xte.shape[3]
    x_train = idu.reshape_pixel_data(Xtr)
    x_val = idu.reshape_pixel_data(Xte)


    print "Loaded and preprocessed the data.. Ready for training and testing models.."

    return (x_train, x_val, y_train, y_val)


def get_balanced_data(image_res_dir, test_size=0.2):
    cs.updateFileNames(image_res_dir, crop_image=False, sort=False)
    print " cs.IMAGE_RESOLUTION_DIR : " + cs.IMAGE_RESOLUTION_DIR[:-1]

    Xtr, Xte, y_train, y_val = idu.get_image_array_balanced_data(test_size=test_size)
    print(Xtr.shape, Xte.shape)
    # x_train = Xtr  # Xtr_rows becomes 50000 x 3072
    # x_val = Xte  # Xte_rows becomes 10000 x 3072

    assert Xtr.shape[1] == Xte.shape[1] and Xtr.shape[2] == Xte.shape[2] and Xtr.shape[3] == Xte.shape[3]
    x_train = idu.reshape_pixel_data(Xtr)
    x_val = idu.reshape_pixel_data(Xte)

    print "Loaded and preprocessed the data.. Ready for training and testing models.."

    return (x_train, x_val, y_train, y_val)


def test_model(crop_image, image_res_dir):
    #    print ("Directory: ", image_res_dir)
    print (
        "ImageResolution\tModel\tCV\tAcc\tPrec_ACUTE\tPrec_CHRONIC\tRecall_ACUTE\tRecall_CHRONIC\tF1_score_ACUTE\tF1_score_CHRONIC\tSupport_ACUTE\tSupport_CHRONIC\t AUC\tparams")
    test_size_split_ratio = 0.2
    '''
    #for i in range(1,9):
    	#x_train, x_val, y_train, y_val = get_data(crop_image, image_res_dir, test_size=test_size_split_ratio)
    	x_train, x_val, y_train, y_val = get_balanced_data(image_res_dir, test_size=test_size_split_ratio)

  	params={'cv':True, 'dataset':'balanced', 'trail': str(i)}
 	custom_nn_test(x_train, x_val, y_train, y_val, 'lsvm', params)

    '''

    x_train, x_val, y_train, y_val = get_data(crop_image, image_res_dir, test_size=test_size_split_ratio)
    params={'cv':True, 'dataset':'roc'}
    custom_nn_roc(x_train, x_val, y_train, y_val, 'svc_linear', params)

    #params = {'cv': True, 'test_size': test_size_split_ratio}
    # custom_nn_test(x_train, x_val, y_train, y_val, 'dt', params)
    #custom_nn_test(x_train, x_val, y_train, y_val, 'lsvm', params)
    # custom_nn_test(x_train, x_val, y_train, y_val, 'svc_linear', params)

    # custom_nn_test(x_train, x_val, y_train, y_val, 'svc_rbf', params)


def grid_param_search(crop_image, image_res_dir):
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
    print ("###################################################### ")
    x_train, x_val, y_train, y_val = get_data(crop_image, image_res_dir)
    x_all = np.concatenate((x_train, x_val), axis=0)
    y_all = np.concatenate((y_train, y_val), axis=0)

    C_range = [0.5, 1, 2, 3, 5, 10, 20]  # np.logspace(-2, 4, 13)
    param_grid = dict(C=C_range)
    svc = svm.SVC(kernel='linear')
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    print ("Starting grid search")
    grid = GridSearchCV(svc, param_grid=param_grid, cv=cv)
    grid.fit(x_all, y_all)
    print ("Crop data : ", crop_image, ", C_range = [0.5,1,2,3,5,10,20 ]")
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    print ("ALL CV RESULTS: \n", grid.cv_results_)


def run_models(crop_image, image_res_dir):
    # print ("Directory: ", image_res_dir)
    print (
        "ImageResolution\tModel\tCV\tAcc\tPrec_ACUTE\tPrec_CHRONIC\tRecall_ACUTE\tRecall_CHRONIC\tF1_score_ACUTE\tF1_score_CHRONIC\tSupport_ACUTE\tSupport_CHRONIC\t AUC\tparams")
    test_size_split_ratio = 0.2

    x_train, x_val, y_train, y_val = get_data(crop_image, image_res_dir, test_size=test_size_split_ratio)

    params = {'crop_image': crop_image, 'cv': False, 'test_size': test_size_split_ratio}
    custom_nn_test(x_train, x_val, y_train, y_val, 'svc_linear', params)
    custom_nn_test(x_train, x_val, y_train, y_val, 'lsvm', params)

    types = ["dt", "random_forest", "guassian_nb", "SGD"]
    for type in types:
        custom_nn_test(x_train, x_val, y_train, y_val, type, params)

    distances = ['l1', 'minkowski', 'manhattan']
    for dist in distances:
        for k in [1, 3, 5, 9, 10, 15, 20]:
            params['k'] = k
            params['distance'] = dist
            custom_nn_test(x_train, x_val, y_train, y_val, 'scikit_knn', params)

        ############### CV: False ##########
        params = {'crop_image': crop_image, 'cv': True}
    custom_nn_test(x_train, x_val, y_train, y_val, 'svc_linear', params)
    custom_nn_test(x_train, x_val, y_train, y_val, 'lsvm', params)

    types = ["dt", "random_forest", "guassian_nb", "SGD"]
    for type in types:
        custom_nn_test(x_train, x_val, y_train, y_val, type, params)

    distances = ['l1', 'minkowski', 'manhattan']
    for dist in distances:
        for k in [1, 3, 5, 9, 10, 15, 20]:
            params['k'] = k
            params['distance'] = dist
            custom_nn_test(x_train, x_val, y_train, y_val, 'scikit_knn', params)


if __name__ == '__main__':
    test_model(False, 'img_480_480')

    #crop_image = False
    #for image_res_dir in ['img_600_600', 'img_650_650', 'img_700_700', 'img_750_750', 'img_800_800', 'img_850_850',
    #                      'img_900_900', 'img_950_950', 'img_1000_1000']:
        #
        # ['img_50_50','img_100_100','img_150_150','img_200_200','img_250_250','img_300_300','img_350_350','img_400_400','img_450_450','img_500_500','img_550_550']:         #grid_param_search(crop_image, image_res_dir)
        # run_models(crop_image, image_res_dir)
        #test_model(crop_image, image_res_dir)
        # print image_res_dir
