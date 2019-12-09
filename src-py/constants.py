#Labels of the data
CHRONIC_LABEL=1
CLINIC_LABEL=-1
CHRONIC_LABEL_STRING="Chronic"
CLINIC_LABEL_STRING="Recent"


DATA_DIR="../data/"
RAW_DATA_DIR_NAME="orig_data/"
CHRONIC_DIR_NAME="Chronics_NGS"
CLINIC_DIR_NAME="Acutes_NGS2"
IMAGE_ROOT_DATA_DIR= DATA_DIR+"img_data/"
IMAGE_RESOLUTION_DIR="img_480_480/"
IMAGE_DATA_DIR= IMAGE_ROOT_DATA_DIR+IMAGE_RESOLUTION_DIR

#IMAGE_DATA_DIR= DATA_DIR+"img_data/img_480_480/"

IMAGE_DATA_DIR_ALL= IMAGE_DATA_DIR +"all/"
#IMAGE_DATA_DIR_ALL= IMAGE_DATA_DIR +"all_crop/"
#IMAGE_DATA_DIR_ALL= IMAGE_DATA_DIR +"all_crop_sort/"
MODEL_PERSIST_PATH="../results/models/"

#CHRONIC_NUM_SEQ=CHRONIC_DIR_NAME+"_num_of_seq.txt"
#CLINIC_NUM_SEQ=CLINIC_DIR_NAME+"_num_of_seq.txt"

def updateFileNames(image_resolution_dir,  crop_image=False, sort=False) : #Eg: updateFileNames(img_480_480 , 18)
	global IMAGE_RESOLUTION_DIR
	global IMAGE_DATA_DIR
	global IMAGE_DATA_DIR_ALL
	
        #IMAGE_RESOLUTION_DIR=str(image_resolution_dir)+'/'
	IMAGE_RESOLUTION_DIR=image_resolution_dir+'/'
	
	IMAGE_DATA_DIR= IMAGE_ROOT_DATA_DIR+IMAGE_RESOLUTION_DIR
	temp="all"
	temp= temp+"_crop" if crop_image else temp
	temp= temp+"_sort" if sort else temp
	
	IMAGE_DATA_DIR_ALL= IMAGE_DATA_DIR +temp+"/"

	print(IMAGE_DATA_DIR_ALL)

##### OUTBREAKS INFORMATION ########
OUTBREAK_IMAGE_RESOLUTION=480
OUTBREAKS_DATA_DIR_NO_XX=DATA_DIR+'outbreaks_data/orig_data/all_except_xx/'
OUTBREAKS_DATA_DIR_WITH_XX=DATA_DIR+'outbreaks_data/orig_data/all_with_xx/'
OUTBREAKS_CONSENSES_DIR_NO_XX=DATA_DIR+'outbreaks_data/consenses_data/all_except_xx/'
OUTBREAKS_CONSENSES_DIR_WITH_XX=DATA_DIR+'outbreaks_data/consenses_data/all_with_xx/'

dirname_temp=str(OUTBREAK_IMAGE_RESOLUTION)+'_'+str(OUTBREAK_IMAGE_RESOLUTION)+'/'

OUTBREAKS_IMG_DIR_NO_XX=DATA_DIR+'outbreaks_data/outbreaks_img_data/img_'
OUTBREAKS_IMG_DIR_WITH_XX=DATA_DIR+'outbreaks_data_with_xx/outbreaks_img_data/img_'

#OUTBREAKS_IMG_DIR=DATA_DIR+'outbreaks_data/outbreaks_img_data/img_480_480/'
#OUTBREAKS_IMG_DIR=DATA_DIR+'outbreaks_data/outbreaks_img_data/img_512_512/'

OUTBREAKS_IMG_DIR=OUTBREAKS_IMG_DIR_NO_XX+dirname_temp
OUTBREAKS_IMG_DIR_ALL=OUTBREAKS_IMG_DIR +"all/"
OUTBREAKS_IMG_DIR_ALL_CROP= OUTBREAKS_IMG_DIR +"all_crop/"
OUTBREAKS_IMG_DIR_ALL_CROP_SORT= OUTBREAKS_IMG_DIR +"all_crop_sort/"

 #Eg: updateFileNames_Outbreaks(480, crop_image=True, sort=False, withXX=True)
def updateFileNames_Outbreaks(image_resolution,  crop_image=False, sort=False, withXX=False) :
	global OUTBREAK_IMAGE_RESOLUTION
	global OUTBREAKS_DATA_DIR
	global OUTBREAKS_IMG_DIR
	global OUTBREAKS_IMG_DIR_ALL
	global OUTBREAKS_IMG_DIR_ALL_CROP
	global OUTBREAKS_IMG_DIR_ALL_CROP_SORT
	
	OUTBREAK_IMAGE_RESOLUTION=image_resolution
        dirname_temp=str(OUTBREAK_IMAGE_RESOLUTION)+'_'+str(OUTBREAK_IMAGE_RESOLUTION)+'/'

	OUTBREAKS_DATA_DIR=OUTBREAKS_DATA_DIR_WITH_XX  if withXX else OUTBREAKS_DATA_DIR_NO_XX
	OUTBREAKS_IMG_DIR=OUTBREAKS_IMG_DIR_WITH_XX+dirname_temp  if withXX else OUTBREAKS_IMG_DIR_NO_XX+dirname_temp


	temp="all"
	OUTBREAKS_IMG_DIR_ALL= OUTBREAKS_IMG_DIR +temp+"/"

	temp= temp+"_crop" if crop_image else temp
	OUTBREAKS_IMG_DIR_ALL_CROP= OUTBREAKS_IMG_DIR +temp+"/"

	temp= temp+"_sort" if sort else temp
	OUTBREAKS_IMG_DIR_ALL_CROP_SORT= OUTBREAKS_IMG_DIR +temp+"/"
	

	#print OUTBREAKS_IMG_DIR_ALL


#This stores in dict format the cluster ids along with the number of samples in each cluster
OUTBREAKS_ACTUAL_CLUSTERS="{'AA' : 3, 'AB' : 2, 'AC' : 4, 'AD' : 2, 'AE' : 3, 'AH' : 2, 'AI' : 15, 'AJ' : 3, 'AK' : 3, 'AL' : 2, 'AN' : 2, 'AO' : 4, 'AQ' : 9, 'AR' : 4, 'AS' : 2, 'AU' : 2, 'AV' : 2, 'AW' : 19, 'AX' : 11, 'AY' : 3, 'BA' : 6, 'BB' : 7, 'BC' : 2, 'BD' : 4, 'BE' : 4, 'BI' : 4, 'BJ' : 4, 'BK' : 3, 'BN' : 2, 'BP' : 2, 'BT' : 2, 'BU' : 2, 'BX' : 3}"

