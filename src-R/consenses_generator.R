#!/usr/bin/env Rscript

#Generates a consenses sequence from a fasta file i.e., generating a entire fasta file to a single sequence. This is done by generating a new sequence by replacing each column of nucleotides with its most frequent nucleotide. In case of tie, we can replace any from the tie-set.
args = commandArgs(trailingOnly=TRUE)

require(alignfigR)
library(ggplot2)

label="consensus"

all_dir_name="all_with_xx"

path =paste( "../data/outbreaks_data/orig_data",all_dir_name, sep="/")
out_path=paste("../data/outbreaks_data/consenses_data",all_dir_name, sep="/")

#create dir if it doesnot exist
dir.create(out_path, recursive = TRUE)

file.names <- dir(path, pattern =".fas")
for(i in 1:length(file.names)){
    my_data <- read_alignment(file.path(path, file.names[i]))
    df=as.data.frame(my_data)

    out_file_name=file.path(out_path,paste(label, file.names[i],length(my_data),".txt", sep = "_"))
    print (paste("Generating consenses for file",out_file_name ))
    #Incorrect -- consenses_data=paste(sapply(my_data,function(x) names(which.max(table(x)))), collapse="")
    consenses_data=paste(apply(df,1,function(x) names(which.max(table(x)))), collapse="")
    write(consenses_data, file=out_file_name)
}
