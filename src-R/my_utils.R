#!/usr/bin/env Rscript

## Modified original source taken from https://github.com/sjspielman/alignfigR/blob/master/R/alignfigR.R

#Removes any sequence in the file which has non-standard length
my_read_alignment <- function(file){
    raw_data <- readLines( file, warn = FALSE )
    seq_vector <- c()
    seq_name <- ""
    for (line in raw_data){
        # New sequence record? Reset numbering
        if ( grepl("^>", line) ){
            seq_name <- sub("^>", "", line)
            seq_vector[seq_name] <- ""
        }
        else {
            temp_seq <- gsub(" ","",line)
            temp_seq <- gsub("\n","",temp_seq)
            seq_vector[seq_name] <- paste( seq_vector[seq_name], temp_seq, sep="" )
        }
    }
    # Is this an alignment?
    seq_list <- strsplit(seq_vector, split = "")
    lengths <- sapply(seq_list, length)
    if ( sum(lengths != lengths[1]) != 0 ){
        print(paste("Your provided file is not an alignment. Removing sequence with non-standard length. FileName: ", file))
	print(paste("BP Length: ", lengths[1]))
	p<-(lengths == lengths[1])
        seq_list <- seq_list[p]
    }
    # Return sequence data parsed into named list
    seq_list 
}
