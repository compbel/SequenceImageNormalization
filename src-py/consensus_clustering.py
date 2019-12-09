# use 'Agg' backend to run from command line and write to a file
import matplotlib
from setuptools.command.dist_info import dist_info

matplotlib.use('Agg')

import numpy as np
import image_data_utils as idu
import constants as cs
from scipy.spatial import distance
#from leven import levenshtein
import pandas as pd


x_data=[]

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def hamming_metric(x, y):
    i, j = int(x[0]), int(y[0]) # extract indices
    return hamming2(x_data[i], x_data[j])
'''
def lev_metric(x, y):
    i, j = int(x[0]), int(y[0]) # extract indices
    return levenshtein(x_data[i], x_data[j])
'''

def get_alg_hier_clustering(num_clusters, params):
    import scipy.cluster.hierarchy as sch
    from sklearn.cluster import AgglomerativeClustering

    # return AgglomerativeClustering(n_clusters=num_clusters, linkage = params['linkage'])
    #params['affinity']=params['metric']
    return AgglomerativeClustering(n_clusters=num_clusters, affinity=params['affinity'], linkage=params['linkage'])

def get_DBSCAN(num_clusters, params):
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=0.001, min_samples=params['n_init'], metric=params['affinity'])

#TODO works with pariwise distance; to check
#def get_affinity_propagation(num_clusters, params):
    #from sklearn.cluster import AffinityPropagation

    #return AffinityPropagation();


def get_feature_agglomeration(num_clusters, params):
    from sklearn.cluster import FeatureAgglomeration

    return FeatureAgglomeration(n_clusters=num_clusters, affinity=params['affinity'], linkage=params['linkage']);


def get_model(type, params):
    if type == "agglom_clus":
        model = get_alg_hier_clustering(params['num_clusters'], params)
    elif type == "db_scan":
        model = get_DBSCAN(params['num_clusters'], params)
    elif type == "feature_agglo":
        model = get_feature_agglomeration(params['num_clusters'], params)
    else:
        model = None

    return model

def get_pairwise_distance(x_data):
    n = len(x_data)

    my_array = np.zeros((n,n))
    #
    for i, ele_1 in enumerate(x_data):
        for j, ele_2 in enumerate(x_data):
            if j >= i:
                break # Since the matrix is symmetrical we don't need to
                      # calculate everything
            dist = hamming_metric([i],[j])
            my_array[i, j] = dist
            my_array[j, i] = dist

    return my_array


def get_outbreaks_consenses_data( withXX=False):
    from scipy.spatial.distance import pdist
    global x_data
    x_data, y_actual, file_names = idu.get_consensus_array_data(withXX)
    print "Loaded and preprocessed the data.. Ready for training and testing models.."
    #pairwise_distances=distance.pdist(x_data, hamming_metric)
    pairwise_distances=get_pairwise_distance(x_data)
    x_data=x_data.reshape(-1, 1)
    return (x_data, y_actual, file_names, pairwise_distances)


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


def get_relatedness_measure(withXX=True):
    out_filename = '../outbreaks_consenses_' +  label_for_xx+ '.csv'
    distance_array = [ 'hamming']  # , 'seuclidean', 'wminkowski']'mahalanobis',
    #distance_array = ['levenshtein', 'hamming']  # , 'seuclidean', 'wminkowski']'mahalanobis',
    header_arr = ['outbreak_fileid1', 'outbreak_fileid2', 'label1', 'label2', 'label_compare']
    header_arr.extend(distance_array)
    df = ''
    try:
        df = pd.read_csv(out_filename)

    except:
        print "Calculating Distances..."
        x_data, y_actual, file_names, pdist = get_outbreaks_consenses_data(withXX=withXX)
        data = [];

        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                are_same = False
                bn1 = file_names[i].split("_")[1];
                bn2 = file_names[j].split("_")[1];

                ham= pdist[i][j];
                #leven= lev_metric([i], [j])

                if y_actual[i] != 'XX' and y_actual[j] != 'XX':
                    are_same = y_actual[i] == y_actual[j]

                arr = [bn1, bn2, y_actual[i], y_actual[j], are_same, ham, ]#leven]  # , seuclidean, wminkowski , mahalanobis]
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

        #print false_idx[0][0],df1['label_compare'][false_idx[0][0]],true_label_col[0:false_idx[0][0]-1]
        print false_idx

        total_true_vals = len(true_idx)
        FNR = (total_true_vals - false_idx[0][0]) / (1. * total_true_vals)
        TPR = (false_idx[0][0]) / (1. * total_true_vals)

        print col_name, FNR, TPR, total_true_vals,false_idx[0][0]


        x, y, auc = get_auc(true_label_col)
        all_x.append(x)
        all_y.append(y)
        all_auc.append(auc)

    plot_auc(all_x, all_y, all_auc, distance_array, withXX);


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
    model.fit(params['pdist'])
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


def plot_auc(all_x, all_y, all_auc, distance_array,withXX):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    type='relatedness_roc_consensus'+ label_for_xx
    p=[];

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


def test_model(withXX):
    x_data, y_actual, file_names, pairwise_distances = get_outbreaks_consenses_data(withXX)

    params = {'type': "agglom_clus", 'num_clusters': 33, 'linkage': 'complete',
              'affinity': 'precomputed', 'metric':hamming_metric, 'pdist':pairwise_distances}
    custom_clustering(x_data, y_actual, file_names, params['type'], params)

    return


def sweep_hier_clustering(withXX):
    x_data, y_actual, file_names, pairwise_distances = get_outbreaks_consenses_data(withXX)

    for linkage in ['complete', 'average', 'single']:

        params = {'type': "agglom_clus", 'num_clusters': 33, 'linkage': linkage,
              'affinity': 'precomputed', 'metric':hamming_metric, 'n_init': 20, 'random_seed': 42,  'pdist':pairwise_distances}
        custom_clustering(x_data, y_actual, file_names, params['type'], params)

        params = {'type': "feature_agglo", 'num_clusters': 33, 'linkage': linkage,
              'affinity': 'precomputed', 'metric':hamming_metric, 'n_init': 20, 'random_seed': 42,  'pdist':pairwise_distances}
        custom_clustering(x_data, y_actual, file_names, params['type'], params)



def run_models(withXX):
    nmi_results = []
    ars_results = []
    x_data, y_actual, file_names, pairwise_distances = get_outbreaks_consenses_data(withXX)

    types = ["agglom_clus", "db_scan","feature_agglo"]
    for type in types:

        params = {'type': type, 'num_clusters': 33, 'linkage': 'single',
              'affinity': 'precomputed', 'metric':hamming_metric, 'pdist':pairwise_distances, 'n_init': 20, 'random_seed': 42}
        metric_names, metric_values = custom_clustering(x_data, y_actual, file_names, params['type'], params)

        ars_results.append(metric_values[0])
        nmi_results.append(metric_values[2])

    return nmi_results, ars_results, types

if __name__ == '__main__':
    withXX=False
    label_for_xx="with_xx" if withXX else "without_xx"
    #test_model(withXX )

    #get_relatedness_measure( withXX=withXX)
    #sweep_hier_clustering(withXX)
    #nmi_results, ars_results, methods=run_models(withXX)
    #plot_results(nmi_results, ars_results, methods)