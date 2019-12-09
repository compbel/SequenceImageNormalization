# use 'Agg' backend to run from command line and write to a file
import matplotlib
from setuptools.command.dist_info import dist_info

matplotlib.use('Agg')

import numpy as np
import image_data_utils as idu
import constants as cs


def get_k_means(num_clusters, params):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=num_clusters, n_init=params['n_init'], random_state=params['random_seed'])


def get_alg_hier_clustering(num_clusters, params):
    import scipy.cluster.hierarchy as sch
    from sklearn.cluster import AgglomerativeClustering

    # return AgglomerativeClustering(n_clusters=num_clusters, linkage = params['linkage'])
    return AgglomerativeClustering(n_clusters=num_clusters, affinity=params['affinity'], linkage=params['linkage'])


def get_PCA(num_clusters, params):
    from sklearn.decomposition import PCA

    return PCA()


def get_ICA(num_clusters, params):
    from sklearn.decomposition import FastICA

    return FastICA()


def get_DBSCAN(num_clusters, params):
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=0.001, min_samples=params['n_init'])


def get_spectral_clustering(num_clusters, params):
    import scipy.cluster.hierarchy as sch
    from sklearn.cluster import SpectralClustering

    return SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_init=params['n_init'],
                              random_state=params['random_seed'], eigen_solver='arpack')


def get_mean_shift(num_clusters, params):
    from sklearn.cluster import MeanShift, estimate_bandwidth
    # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

    return MeanShift(bandwidth=params['bandwidth'], bin_seeding=True)


def get_birch(num_clusters, params):
    from sklearn.cluster import Birch

    return Birch(threshold=0.5, n_clusters=num_clusters)


def get_mini_batch_k_means(num_clusters, params):
    from sklearn.cluster import MiniBatchKMeans

    return MiniBatchKMeans(n_clusters=num_clusters, n_init=params['n_init'], random_state=params['random_seed']);


def get_affinity_propagation(num_clusters, params):
    from sklearn.cluster import AffinityPropagation

    return AffinityPropagation();


def get_feature_agglomeration(num_clusters, params):
    from sklearn.cluster import FeatureAgglomeration

    return FeatureAgglomeration(n_clusters=num_clusters);


def get_model(type, params):
    if type == "k_means":
        model = get_k_means(params['num_clusters'], params)
    elif type == "agglom_clus":
        model = get_alg_hier_clustering(params['num_clusters'], params)
    elif type == "spectral_clus":
        model = get_spectral_clustering(params['num_clusters'], params)
    elif type == "db_scan":
        model = get_DBSCAN(params['num_clusters'], params)
    elif type == "mean_shift":
        model = get_mean_shift(params['num_clusters'], params)
    elif type == "pca":
        model = get_PCA(params['num_clusters'], params)
    elif type == "ica":
        model = get_ICA(params['num_clusters'], params)
    elif type == "birch":
        model = get_birch(params['num_clusters'], params)
    elif type == "mini_batch_k_means":
        model = get_mini_batch_k_means(params['num_clusters'], params)
    elif type == "affinity_prop":
        model = get_affinity_propagation(params['num_clusters'], params)
    elif type == "feature_agglo":
        model = get_feature_agglomeration(params['num_clusters'], params)
    else:
        model = null

    return model


def reshape_pixel_data(img_data):
    return img_data.reshape(img_data.shape[0], img_data.shape[1] * img_data.shape[2] * img_data.shape[3])


def get_outbreaks_data(crop_image=False, sort=False, image_resolution=480, withXX=False):
    cs.updateFileNames_Outbreaks(image_resolution, crop_image=crop_image, sort=sort, withXX=withXX)
    img_data, y_actual, file_names = idu.get_image_array_data(crop_image)

    x_data = reshape_pixel_data(img_data)

    print "Loaded and preprocessed the data.. Ready for training and testing models.."

    return (x_data, y_actual, file_names)


def get_clusters(y_pred_num, y_actual, file_names):
    from collections import defaultdict

    clusters = defaultdict(list)
    cluster_groups = defaultdict(lambda: defaultdict(int))

    for i in range(len(y_pred_num)):
        clus_group = y_actual[i]
        cluster_groups[clus_group][y_pred_num[i]] = cluster_groups[clus_group][y_pred_num[i]] + 1

        clusters[y_pred_num[i]].append(file_names[i].split("_")[1])

    return clusters, cluster_groups;


def print_clusters(cluster_dict):
    for k, v in sorted(cluster_dict.items()):
        print k, " : ", v

    return


def analyse_clusters(cluster_dict):
    from collections import defaultdict

    cluster_seg = defaultdict(lambda: defaultdict(int))

    for k1, filenames in cluster_dict.items():
        for filename_id in filenames:
            clus_group = filename_id[0:2]
            cluster_seg[k1][clus_group] = cluster_seg[k1][clus_group] + 1

    return cluster_seg


def get_performance_metrics(y_pred_num, y_actual):
    from sklearn import metrics
    adjusted_rand_score = metrics.adjusted_rand_score(y_actual, y_pred_num)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(y_actual, y_pred_num)
    normalized_mutual_info_score = metrics.normalized_mutual_info_score(y_actual, y_pred_num)
    homogeneity_score = metrics.homogeneity_score(y_actual, y_pred_num)
    completeness_score = metrics.completeness_score(y_actual, y_pred_num)
    v_measure_score = metrics.v_measure_score(y_actual, y_pred_num)
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(y_actual, y_pred_num)

    metric_names = ["adjusted_rand_score", "adjusted_mutual_info_score", "normalized_mutual_info_score",
                    "homogeneity_score", "completeness_score", "v_measure_score", "fowlkes_mallows_score"];
    metric_values = [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score,
                     completeness_score, v_measure_score, fowlkes_mallows_score]

    print "Metrics"
    print "\t".join(metric_names)
    print('\t'.join([str(x) for x in metric_values]))

    return metric_names, metric_values


def get_relatedness_measure(image_resolution, withXX=True):
    from scipy.spatial import distance
    import pandas as pd

    out_filename = '../outbreaks_distance_res_' + str(image_resolution) + '.csv'
    distance_array = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'euclidean',
                      'minkowski', 'sqeuclidean']  # , 'seuclidean', 'wminkowski']'mahalanobis',
    header_arr = ['outbreak_fileid1', 'outbreak_fileid2', 'label1', 'label2', 'label_compare']
    header_arr.extend(distance_array)
    df = ''
    crop_image = True

    try:
        df = pd.read_csv(out_filename)

    except:
        print "Calculating Distances..."
        x_data, y_actual, file_names = get_outbreaks_data(crop_image=crop_image, sort=False,
                                                          image_resolution=image_resolution, withXX=withXX)
        data = [];

        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                are_same = False
                bn1 = file_names[i].split("_")[1];
                bn2 = file_names[j].split("_")[1];

                braycurtis = distance.braycurtis(x_data[i], x_data[j]);
                canberra = distance.canberra(x_data[i], x_data[j]);
                chebyshev = distance.chebyshev(x_data[i], x_data[j]);
                cityblock = distance.cityblock(x_data[i], x_data[j]);
                correlation = distance.correlation(x_data[i], x_data[j]);
                cosine = distance.cosine(x_data[i], x_data[j]);
                euclidean = distance.euclidean(x_data[i], x_data[j]);
                minkowski = distance.minkowski(x_data[i], x_data[j]);
                sqeuclidean = distance.sqeuclidean(x_data[i], x_data[j]);
                # seuclidean = distance.seuclidean(x_data[i], x_data[j]);
                # wminkowski = distance.wminkowski(x_data[i], x_data[j]);
                # mahalanobis = distance.mahalanobis(x_data[i], x_data[j]);

                if y_actual[i] != 'XX' and y_actual[j] != 'XX':
                    are_same = y_actual[i] == y_actual[j]

                arr = [bn1, bn2, y_actual[i], y_actual[j], are_same, braycurtis, canberra, chebyshev, cityblock,
                       correlation, cosine, euclidean, minkowski,
                       sqeuclidean]  # , seuclidean, wminkowski , mahalanobis]
                # Append to data
                data.append(arr)

        # Create panda dataframe
        df = pd.DataFrame(data, columns=header_arr)

        # Write to CSV
        df.to_csv(out_filename)

    # Sort df based on different distance keys and then print ratios
    all_x = []
    all_y = []
    all_auc = []

    for dist_indx in range(len(header_arr) - len(distance_array), len(header_arr)):
        col_name = header_arr[dist_indx]
        is_ascending = True  # if  col_name!='correlation' else False
        df1 = df.sort_values(by=col_name, ascending=is_ascending);
        true_idx = df1.index[df1['label_compare'] == True]
        true_label_col = np.asarray(list(df1['label_compare']))
        false_idx = np.where(true_label_col == False)

        # print false_idx[0][0],df1['label_compare'][false_idx[0][0]],true_label_col[0:false_idx[0][0]-1]

        total_true_vals = len(true_idx)
        FNR = (total_true_vals - false_idx[0][0]) / (1. * total_true_vals)
        TPR = (false_idx[0][0]) / (1. * total_true_vals)

        print col_name, FNR, TPR, total_true_vals,false_idx[0][0]


        x, y, auc = get_auc(true_label_col)
        all_x.append(x)
        all_y.append(y)
        all_auc.append(auc)

    plot_auc(all_x, all_y, all_auc, distance_array, image_resolution);


    return;


def get_auc(true_label_col):
    from sklearn import metrics
    x = []
    y = []
    trueCount = 0;
    falseCount = 0;

    for val in true_label_col:
        if (val):
            trueCount = trueCount + 1
        else:
            falseCount = falseCount + 1
        x.append(falseCount)
        y.append(trueCount)

    x = np.asarray(x)/(1.*falseCount)
    y = np.asarray(y)/(1.*trueCount)

    return x, y, metrics.auc(x, y)


def custom_clustering(x_data, y_actual, file_names, type, params):
    print "\n############################\nType: ", type
    print "params", params
    model = get_model(type, params)
    print model
    model.fit(x_data)
    # y_pred_num=model.predict(x_data)
    y_pred_num = list(model.labels_)

    print y_pred_num
    print "y_actual", y_actual

    pred_clusters, pred_cluster_groups = get_clusters(y_pred_num, y_actual, file_names)
    # actual_clusters=get_clusters(
    # print pred_clusters
    print_clusters(pred_clusters)
    print_clusters(pred_cluster_groups)
    print_clusters(analyse_clusters(pred_clusters))
    metric_names, metric_values = get_performance_metrics(y_pred_num, y_actual)

    return metric_names, metric_values


def plot_auc(all_x, all_y, all_auc, distance_array, image_resolution):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    type='relatedness_roc_'+str(image_resolution)
    p=[];

    print "removed ", distance_array.pop(2)
    all_auc.pop(2)
    all_x.pop(2)
    all_y.pop(2)

    orderedListIndx=sorted(range(len(all_auc)), key=lambda k: all_auc[k], reverse=True)

    for i in orderedListIndx:
      # Plot the number in the list and set the line thickness.
      p.append( plt.plot(all_x[i], all_y[i]))

    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both', labelsize=9)

    #plt.xticks(x_vals[0:11], xtic_labels[0:11])

    plt.legend([ distance_array[i]+ " " + str(round(all_auc[i], 3))  for i in orderedListIndx])

    #plt.xticks(rotation=45)

    ymin,ymax=plt.ylim()
    plt.ylim(ymin=0.7, ymax=1.01)
    plt.xlim(xmin=0., xmax=0.3)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Display the plot in the matplotlib's viewer.
    plt.show()

    plt.savefig(type+'.pdf', bbox_inches='tight')

    # save the final plot

    plt.savefig("../"+type+".png", dpi=600)
    plt.savefig("../"+type+".pdf", dpi=150)
    #plt.show()

    return


def test_model():
    # params={'type':'k_means', 'num_clusters':10, 'crop_image':False}
    # params={'type':'k_means', 'num_clusters':33, 'crop_image':False}

    # x_data, y_actual, file_names = get_outbreaks_data(params['crop_image'])
    # custom_clustering(x_data, y_actual, file_names , params['type'], params)

    crop_image = True
    x_data, y_actual, file_names = get_outbreaks_data(crop_image)

    from sklearn.cluster import MeanShift, estimate_bandwidth
    # bandwidth = estimate_bandwidth(x_data, quantile=0.2)

    # params={'type':'k_means', 'num_clusters':33, 'crop_image':True, 'n_init':20, 'random_seed':42}

    # params={'type':"agglom_clus", 'num_clusters':33, 'crop_image':crop_image, 'linkage':'ward', 'affinity':"precomputed", 'n_init':20, 'random_seed':42, 'bandwidth' : bandwidth}
    params = {'type': "agglom_clus", 'num_clusters': 33, 'crop_image': crop_image, 'linkage': 'ward',
              'affinity': 'euclidean'}
    custom_clustering(x_data, y_actual, file_names, params['type'], params)

    return


def sweep_hier_clustering(crop_image=True):
    x_data, y_actual, file_names = get_outbreaks_data(crop_image)
    # 'linkage':'ward' works only with affinity: 'euclidean',so testing separately
    params = {'type': "agglom_clus", 'num_clusters': 33, 'crop_image': crop_image, 'linkage': 'ward',
              'affinity': 'euclidean'}
    custom_clustering(x_data, y_actual, file_names, params['type'], params)

    # for linkage in ['ward', 'complete', 'average', 'single'] :
    for linkage in ['complete', 'average']:
        for affinity in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']:
            params = {'type': "agglom_clus", 'num_clusters': 33, 'crop_image': crop_image, 'linkage': linkage,
                      'affinity': affinity}
            # params['affinity']=affinity
            # params['linkage']=linkage
            custom_clustering(x_data, y_actual, file_names, params['type'], params)


def sweep_k_means(crop_image=True):
    x_data, y_actual, file_names = get_outbreaks_data(crop_image)

    for type in ['k_means', 'mini_batch_k_means']:
        for n_init in [5, 10, 20, 30]:
            for random_seed in [42, 66, 437, 8881]:
                params = {'type': type, 'num_clusters': 33, 'crop_image': crop_image, 'n_init': n_init,
                          'random_seed': random_seed}
                custom_clustering(x_data, y_actual, file_names, params['type'], params)


def run_models(crop_image=True):
    nmi_results = []
    ars_results = []
    x_data, y_actual, file_names = get_outbreaks_data(crop_image)

    # types=[  "k_means", "agglom_clus", "spectral_clus", "db_scan", "mean_shift", "pca", "ica" , "mini_batch_k_means", "birch", "affinity_prop" ]
    types = ["k_means", "agglom_clus", "mini_batch_k_means", "birch", "affinity_prop", "feature_agglo"]
    for type in types:
        params = {'type': type, 'num_clusters': 33, 'crop_image': crop_image, 'linkage': 'ward',
                  'affinity': "precomputed", 'n_init': 20, 'random_seed': 42}
        metric_names, metric_values = custom_clustering(x_data, y_actual, file_names, params['type'], params)

        ars_results.append(metric_values[0])
        nmi_results.append(metric_values[2])

    return nmi_results, ars_results, types


'''
Notes:

CROP_IMAGE=TRUE gives better clustering than the other. This might be because the image will not contain axis labels
'''
if __name__ == '__main__':

    get_relatedness_measure(480, withXX=True)
    #get_relatedness_measure(512, withXX=True)
    #get_relatedness_measure(1024, withXX=True)
    # test_model()
    # sweep_hier_clustering()
    # sweep_k_means()
    # nmi_results, ars_results, methods=run_models()
    # plot_results(nmi_results, ars_results, methods)