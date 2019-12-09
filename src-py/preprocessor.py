import pandas as pd
import  glob
import constants as cs


def get_transformed_data(datapath, type):
    data=[]
    label=[]
    filenames=glob.glob(datapath+'/*.fas')
    for fas_file in filenames :
        dna_vir_all=[]
        dna_vir=[]
        lines= open(fas_file, "r").read().splitlines()
        for line in lines :
            if not line.strip() : continue;
            if line.startswith(">") :
                dna_vir_all.append(''.join(dna_vir))
                dna_vir = []
            else:
                dna_vir.extend(line)
        if len(dna_vir) > 0:
            dna_vir_all.append(''.join(dna_vir))
            dna_vir = []

        data.append(cs.SEPARATOR.join( dna_vir_all[1:]))
        label.append(str(type))

    return data, label

def preprocess_data(datapath, outdatapath) :
    out_data_file = open( outdatapath + cs.TRAIN_DATA_FILE_NAME, 'w+')
    out_label_file = open(outdatapath  + cs.TRAIN_DATA_LABEL_FILE_NAME, 'w+')
    for type in [cs.CHRONIC_LABEL, cs.CLINIC_LABEL] :
        if type == cs.CHRONIC_LABEL:
            dirname = cs.CHRONIC_DIR_NAME
        else:
            dirname = cs.CLINIC_DIR_NAME

        data, label = get_transformed_data(datapath+dirname, type)
        out_data_file.write("\n".join(data) +"\n")
        out_label_file.write("\n".join(label)+"\n")

    out_data_file.close()
    out_label_file.close()

if __name__ == "__main__":
    __author__ = 'Sunitha Basodi'
    preprocess_data(cs.DATA_DIR+cs.RAW_DATA_DIR_NAME, cs.DATA_DIR)




