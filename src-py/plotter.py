import  matplotlib as mpl
mpl.use('agg')  ## agg backend is used to create plot as a .png file
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os, glob

matplotlib.rcParams['savefig.dpi'] = 600 #2*sweepSize/figSize #300
defaultFontSize = 8; largeFontSize = 20; indexFontSize = 18;
matplotlib.rcParams.update({'font.size': largeFontSize,
              'axes.labelsize': largeFontSize,
              'axes.titlesize': largeFontSize,
              'font.size': largeFontSize,
              'legend.fontsize': indexFontSize,
              'xtick.labelsize': indexFontSize,
              'ytick.labelsize': indexFontSize})

#comment these two for rainbow text function to run properly, along with setting fig.dpi = 1200 described near rainbow text
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

bbox_props = dict(boxstyle="square,pad=0.2", fc="white", ec="k", lw=.5, alpha=0.)
arrowprops = dict(arrowstyle="-", connectionstyle="arc3", facecolor='white', lw=1.,
                  ec="white")  # ,shrink=0.05, width=defaultFontSize/10.)
bbox_props_fig = dict(boxstyle="square,pad=0.001", fc="white", ec="k", lw=0)
bbox_props_index = dict(boxstyle="square,pad=0.", fc="white", ec="k", lw=0.)
bbox_props_label = dict(boxstyle="square,pad=0.01", fc="white", ec="k", lw=0	)


label_dict={'SGD':'SGD', 'dt':'Decision Tree', 'guassian':'Guassian NB', 'lsvm':'Linear SVM', 'random':'Random Forest', 'scikit':'kNN\_Mink\_15'}

datapath='/home/sbasodi1/Desktop/rm_later/'
datapath='/home/sbasodi1/MEGA/GSU/CSE/lab/collaboration/adv_bio/chr_vs_clinic/data/box_plot_files/all_freq_sorted/'
datapath='/home/sbasodi1/MEGA/GSU/CSE/lab/collaboration/adv_bio/chr_vs_clinic/paper/rm_later/'

os.chdir(datapath)
def get_all_data(datapath, index, printMetrics=False):

    acc_data=[]
    files_info=[]
    method_names=[]
    os.chdir(datapath)
    files = sorted(glob.glob('_type*.csv'))
    #print files
    for filename in files:

        file_info=filename.split("_")
        data=np.loadtxt(filename, delimiter=',', comments='_' )
        #print data , len(data)
        acc_data.append(data[index])

        files_info.append(filename)
        method_names.append(file_info[2])
        if printMetrics:
            get_data_metrics(data, filename)
    #print "\n".join(files)
    return acc_data, method_names#, files_info

def box_plot(data_to_plot) :
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    # Save the figure
    fig.savefig('fig1.png', bbox_inches='tight')
    #plt.show()

def fancy_box_plot(data_to_plot, labels, type):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#FF0000', linewidth=2)

    ## change color and linewidth of the medians
    for mean in bp['means']:
        mean.set(color='#b2df8a', linewidth=2)


    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ## Custom x-axis labels
    ax.set_xticklabels(labels)


    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set

    plt.xticks(rotation=45)

    # Save the figure
    fig.savefig('fancy_fig1_'+type+'.pdf', bbox_inches='tight')
    #plt.show()
    ymin,ymax=plt.ylim()
    plt.ylim(ymax=max(1.01, ymax))
    plt.clf()

def filter_data(data, methods) :
    data_to_plot=data[:]
    labels=[]
    for method in methods:
        labels.append(label_dict[method])

    return data_to_plot, labels


def get_data_metrics(data, label):
    data_mat=np.matrix(data)
    print  "\n#########\nType: ", label
    print "Mean: \n", data_mat.mean(1)
    print "Std Dev: \n", data_mat.std(1)
    #print "Mean , StandardDeviation"
    #print np.hstack((data_mat.mean(1), data_mat.std(1)))

def plot_ml_model_metrics():
    all_types = ["acc", "prec_clinic", "recall_clinic", "f1_score_clinic", "support_clinic",
             "prec_chronic", "recall_chronic", "f1_score_chronic", "support_chronic", "auc"]
    #type = 'auc'
    #index = 9
    for index, curr_type in enumerate(all_types):
        #print(index, curr_type)
        out=get_all_data(datapath, index, printMetrics=True if index==0 else False)
        #box_plot(out[0])
        data_to_plot, labels=filter_data(out[0], out[1])
        #print data_to_plot, labels
        fancy_box_plot(data_to_plot, labels, curr_type)

    return

# Plot a line based on the x and y axis value list.
def draw_line(type, x_number_values, y_number_values, xtic_labels, ylabel):

    # Plot the number in the list and set the line thickness.
    p1= plt.plot(x_number_values, y_number_values, linewidth=3)
    p2= plt.plot(x_number_values, y_number_values, linewidth=3)
    p3= plt.plot(x_number_values, y_number_values, linewidth=3)

    # Set the line chart title and the text font size.
    #plt.title("Square Numbers", fontsize=19)

    # Set x axes label.
    #plt.xlabel(xlabel, fontsize=10)

    # Set y axes label.
    #plt.ylabel(ylabel, fontsize=10)

    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both')

    plt.xticks(xtic_labels)
    plt.legend((p1[0], p2[0], p3[0]), ('NMI', 'Homogeneity', 'Completeness'))

    # Display the plot in the matplotlib's viewer.
    plt.show()

    plt.savefig(type+'.pdf', bbox_inches='tight')
    plt.clf()


def plot_clustering_metrics_ROC(image_resolution=480):
    import image_clustering as ic
    import pandas as pd

    out_filename = 'outbreaks_distance_res_' + str(image_resolution) + '.csv'
    distance_array = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'euclidean',
                      'minkowski', 'sqeuclidean']  # , 'seuclidean', 'wminkowski']'mahalanobis',
    header_arr = ['outbreak_fileid1', 'outbreak_fileid2', 'label1', 'label2', 'label_compare']
    header_arr.extend(distance_array)
    df = ''
    crop_image = True
    df = pd.read_csv(out_filename)

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


        x, y, auc = ic.get_auc(true_label_col)
        all_x.append(x)
        all_y.append(y)
        all_auc.append(auc)

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
    plt.tick_params(axis='both')

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

    plt.savefig("../"+type+".png")
    plt.savefig("../"+type+".pdf")
    plt.clf()
    #plt.show()

    return


def plot_clustering_metrics():
    xtic_labels = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000']
    print xtic_labels[0:11]
    x_vals=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    nmi_vals = [0.96, 0.985, 0.987, 0.976, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987]
    homo_vals = [0.967, 0.992, 0.994, 0.987, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994]
    compl_vals = [0.953, 0.979, 0.979, 0.965, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979]
    type='clustering_resolution'


    # Plot the number in the list and set the line thickness.
    p1= plt.plot(x_vals[0:11], nmi_vals[0:11], 'r-*', linewidth=2)
    p2= plt.plot(x_vals[0:11], homo_vals[0:11], 'b-^', linewidth=2)
    p3= plt.plot(x_vals[0:11], compl_vals[0:11], 'g-s', linewidth=2)

    #p1= plt.plot(x_vals, nmi_vals, linewidth=3)
    #p2= plt.plot(x_vals, homo_vals, linewidth=3)
    #p3= plt.plot(x_vals, compl_vals, linewidth=3)

    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both')

    plt.xticks(x_vals[0:11], xtic_labels[0:11])
    plt.legend((p1[0], p2[0], p3[0]), ('NMI', 'Homogeneity', 'Completeness'))

    plt.xticks(rotation=45)

    ymin,ymax=plt.ylim()
    plt.ylim(ymin=0.95, ymax=max(1, ymax))

    # Display the plot in the matplotlib's viewer.
    plt.show()

    plt.savefig(type+'.pdf', bbox_inches='tight')
    plt.clf()

def plot_classification_metrics():
    xtic_labels = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000']
    x_vals=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    x_vals=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    acc = [0.8496739597, 0.8646776403, 0.931668811668812, 0.967033462033462, 0.96996567996568, 0.966954096954097, 0.967187902187902, 0.972822822822823, 0.981006006006006, 0.96981123981124, 0.972743457743458]
    prec_cli = [0.7247619048, 0.80753988895, 0.865308025308025, 0.945151515151515, 0.957575757575757, 0.972575757575758, 0.963484848484849, 0.973484848484848, 0.990909090909091, 0.955151515151515, 0.982575757575758]
    prec_chro= [0.9137608696, 0.897336670783333, 0.965350427350428, 0.976911680911681, 0.977307285307285, 0.966851753610374, 0.970709303467924, 0.973899877899878, 0.977757427757428, 0.977307285307285, 0.970709303467924]
    rec_cli = [0.8063636364, 0.8711836516, 0.918181818181818, 0.944545454545455, 0.944545454545455, 0.916363636363636, 0.927272727272727, 0.936363636363637, 0.945454545454546, 0.944545454545455, 0.926363636363636]
    rec_chro= [0.8678461538, 0.863742994658333, 0.937692307692308, 0.976615384615385, 0.980615384615385, 0.988307692307692, 0.984307692307692, 0.988307692307692, 0.996153846153846, 0.980461538461539, 0.992307692307692]
    auc= [0.8371048951, 0.865988311441666, 0.927937062937063, 0.96058041958042, 0.96258041958042, 0.952335664335664, 0.95579020979021, 0.962335664335664, 0.970804195804196, 0.962503496503497, 0.959335664335664]
    type='classification_resolution'


    # Plot the number in the list and set the line thickness.
    p1= plt.plot(x_vals, acc, '-*', linewidth=2)
    p2= plt.plot(x_vals, prec_cli, '-^', linewidth=2)
    p3= plt.plot(x_vals, prec_chro, '-s', linewidth=2)
    p4= plt.plot(x_vals, rec_cli, '-o', linewidth=2)
    p5= plt.plot(x_vals, rec_chro, '-v', linewidth=2)
    p6= plt.plot(x_vals, auc, '-d', linewidth=2)

    #p1= plt.plot(x_vals, nmi_vals, linewidth=3)
    #p2= plt.plot(x_vals, homo_vals, linewidth=3)
    #p3= plt.plot(x_vals, compl_vals, linewidth=3)

    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both')

    plt.xticks(x_vals, xtic_labels)
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('Accuracy', 'Precision\_acute', 'Precision\_chronic', 'Recall\_acute', 'Recall\_chronic', 'AUC'))

    plt.xticks(rotation=45)

    ymin,ymax=plt.ylim()
    plt.ylim(ymin=min(ymin,0.95), ymax=max(1, ymax))

    # Display the plot in the matplotlib's viewer.
    plt.show()

    plt.savefig(type+'.pdf', bbox_inches='tight')
    plt.clf()



def plot_lsvm_roc_kfold():
    import matplotlib.pyplot as plt

    from sklearn.metrics import auc

    fprs=[[0,0,1],[0,0,1],[0,0,1],[0,8.333333e-02,1],[0,0,1],[0,8.333333e-02,1],[0,9.090909e-02,1],[0,1.000000e-01,1],[0,0,1],[0,0,1]]
    tprs=[[0,9.285714e-01,1],[0,9.629630e-01,1],[0,1,1],[0,1,1],[0,9.629630e-01,1],[0,1,1],[0,9.615385e-01,1],[0,9.230769e-01,1],[0,1,1],[0,1,1]]
    aucs=[9.642857e-01,9.814815e-01,1,9.583333e-01,9.814815e-01,9.583333e-01,9.353147e-01,9.115385e-01,1,1]

    for i in range(0, len(aucs)):
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i+1, aucs[i]))

    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    #mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('linearsvm_auroc.pdf', bbox_inches='tight')
    #plt.show()



    return

if __name__ == '__main__':
    #plot_lsvm_roc_kfold()
    plot_ml_model_metrics();
    plot_clustering_metrics()
    plot_classification_metrics()
    plot_clustering_metrics_ROC()

