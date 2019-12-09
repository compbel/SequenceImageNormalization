#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

require(alignfigR)
library(ggplot2)
source("my_utils.R")

if (length(args) < 3) {
  stop("Three arguments must be supplied \n (1) LABEL: Chronics_NGS or Acutes_NGS2 \n(2) IMAGE_RESOLUTION: 100 \n(3) TO_DELETE (sequences with non-standard length): False/True .", call.=FALSE)
} 

#label="Chronics_NGS"
#label="Acutes_NGS2"
#width=100
#height=100
#to_delete="False"
print (paste("Args: ",args[1], args[2], args[3] ))

label=args[1]
width=as.numeric(args[2])
height=as.numeric(args[2])
to_delete=as.logical(tolower(args[3]))

crop_image=FALSE
sort_flag=FALSE
all_dir_name="all"

if (identical(crop_image, TRUE)) {
    all_dir_name=paste(all_dir_name, "crop", sep = "_")
}
if (identical(sort_flag, TRUE)) {
    all_dir_name=paste(all_dir_name, "sort", sep = "_")
}

path = paste("../data/orig_data", label, sep = "/")
image_dir=paste("../data/img_data/img",width, height, sep="_")
### TODO Remove the path lines
#path = "../data/CN/fasta/"
#image_dir=paste("../data/CN/img_data/img1",width, height, sep="_")


#image_dir=paste(image_dir,"", sep="/")
out_path=paste(image_dir,all_dir_name, sep="/")
#lines_file_name=file.path(image_dir,paste(label, "num_of_seq.txt", sep = "_"))
#out_file_name=file.path(out_path,paste(label, "plot_%03d",".png", sep = "_"))
#png(filename=out_file_name)
#png(filename = out_file_name, width = width, height = height)

#create dir if it doesnot exist
dir.create(out_path, recursive = TRUE)

#Empty contents of the file
#cat("", file=lines_file_name)

file.names <- dir(path, pattern =".fas")
for(i in 1:length(file.names)){
    if(to_delete){
      my_data_orig <-  try(my_read_alignment(file.path(path, file.names[i])));
    } else {
      my_data_orig <-  try(read_alignment(file.path(path, file.names[i])));
    }
    if(class(my_data_orig) == "try-error") next;

    if (identical(sort_flag, TRUE)) {
	#Order the named list "my_data_orig" so that we sort the sequences alphabetically
	my_data=my_data_orig[names(sort(sapply(my_data_orig, toString)))]
	#my_data=my_data_orig[names(sort(sapply(my_data_orig, paste0, collapse = "")))]
    } else {
	my_data=my_data_orig
    }
    #Write filename and the number of sequences in that file
    #cat(file.names[i], ",",length(my_data),"\n",file=lines_file_name,append=TRUE)

    out_file_name=file.path(out_path,paste(label, file.names[i],length(my_data),".png", sep = "_"))
    print (paste("Generating image for file",out_file_name ))

    png(filename = out_file_name, width = width, height = height)

    my_plot<-plot_alignment(my_data, "dna")


    if (identical(crop_image, TRUE)) {
    my_plot <- my_plot +
	        #Removes plot axis labels, limits and legend
	        theme(axis.line=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(),
	            axis.ticks=element_blank(),axis.text.x=element_blank(), axis.text.y=element_blank(),
	            legend.position="none") +
	        #Removes white space between plot axis and plotted image
	        scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0))
    } else {
	    my_plot <- my_plot +
	        #Removes plot axis titles and legend, title
	        theme(axis.line=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank(),
	            #axis.ticks=element_blank(),axis.text.x=element_blank(), axis.text.y=element_blank(), #Commented for this
	            legend.position="none") +
	        #Removes white space between plot axis and plotted image
	        scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0))
    }

    #Removes white space outside plot axis
    my_plot <- my_plot +theme( plot.margin=grid::unit(c(0,0,0,0), "cm"))

    print(my_plot)
    dev.off()
}
dev.off()
