# Sequence Image Normalization 
# README #


### What is this repository for? ###
This repo is for project related to bioinformatics domain. Analysis of heterogeneous populations such as viral quasispecies is one of the most challenging bioinformatics problems. This repository contains the code of a pre-processing techique called sequence image normalization which converts fasta sequences into images which are later used for machine learning techniques. 
We solve the following problems:
* Classification of HCV stage infections (Recent and Chronic)
* Detection of viral transmission clusters (or outbreaks)wndemo)

### How do I get set up? ###

* Summary of set up


* Dependencies
Python 2.7 and R 
Following packages are required in R
alignfigR, ggplot2 ( Command to install from R terminal: install.packages("ggplot2"))

* Data
The datasets are owned by CDC, and therefore cannot be deposited to our bitbucket page. It has been published in the following papers [1,2]. The data distribution is guided by the US government regulations, and it can be obtained from CDC upon reasonable request. We have added the mentioning of this fact to the paper.

[1] David S Campo, Guo-Liang Xia, Zoya Dimitrova, Yulin Lin, Joseph C Forbi, Lilia Ganova-Raeva, Lili Punkova,Sumathi Ramachandran, Hong Thai, Pavel Skums, et al.  Accurate genetic detection of hepatitis c virustransmissions in outbreak settings.The Journal of infectious diseases, 213(6):957–965, 2015

[2] James Lara, Mahder Teka, and Yury Khudyakov.  Identification of recent cases of hepatitis c virus infectionusing physical-chemical properties of hypervariable region 1 and a radial basis function neural network classifier.BMC genomics, 18(10):880, 2017.

Please contact Dr. Pavel Skums for any dataset related questions or suggestions.

### Running the code ###
* Get the dataset 
* R scripts (in scr-R directory) are used to convert the sequences to images. 
* Python scripts (in src-py) are used for machine learning models.

### Who do I talk to? ###

* Repo owner or admin: Sunitha Basodi
