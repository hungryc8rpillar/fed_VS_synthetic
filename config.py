#paths
RBC_path_train = '/data/cellface/6_Container/traindata/3types/Separated cells/21-10-22_RBC_M1.seg'
#PLT_path_train = '/data/cellface/6_Container/traindata/3types/Separated cells/03-11-22_Ficoll_PLT_M1.seg'
PLT_path_train = '/data/cellface/6_Container/seg_yolo/25-01-23/PLT_Isolation/PLT-Isolation_M1.seg'

RBC_path_test = '/data/cellface/6_Container/traindata/3types/Separated cells/21-10-22_RBC_M2.seg'
#PLT_path_test = '/data/cellface/6_Container/traindata/3types/Separated cells/03-11-22_Ficoll_PLT_M2.seg'
PLT_path_test = '/data/cellface/6_Container/seg_yolo/25-01-23/PLT_Isolation/PLT-Isolation_M5.seg'

WBC_path_train = '/data/cellface/6_Container/seg_yolo/19-01-23/Neutrophil/19-01-23_Neu_M2_thresh30.seg'
WBC_path_test = '/data/cellface/6_Container/seg_yolo/19-01-23/Neutrophil/19-01-23_Neu_M3.seg'

TCell_path_train = '/data/cellface/6_Container/seg_yolo/10-11-22/10-11-22_MACS/10-11-22_MACS_TCell_M2.seg'
TCell_path_test = '/data/cellface/6_Container/seg_yolo/10-11-22/10-11-22_MACS/10-11-22_MACS_TCell_M1.seg'


CLASS_PATHS_TRAIN = [RBC_path_train,PLT_path_train, WBC_path_train]#, TCell_path_train]
CLASS_PATHS_TEST = [RBC_path_test,PLT_path_test, WBC_path_test]#, TCell_path_test]

CLASS_NAMES = ['RBC','PLT','WBC']

#general variables
USE_CELL_DATA = False
USE_MNIST_DATA = True

if USE_CELL_DATA:
    SAMPLE_LIMIT =  2000
if USE_MNIST_DATA:
    SAMPLE_LIMIT = 5000
BATCH_SIZE = 32
BALANCE = True
if USE_CELL_DATA:
    IMG_SIZE = 96
if USE_MNIST_DATA:
    IMG_SIZE = 28
NUM_PARTITIONS = 2

#classifier variables
CLASSIFIER_EPOCHS = 80
CLASSIFIER_DEVICE = 'cuda'

#privacy variables
DELTA = 1e-3
EPSILON = [16,8,4,2,1]#[5000, 1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05]

#gan variables
GAN_BATCH_SIZE = 8
GAN_EPOCHS = 100
GAN_DEVICE = 'cuda'
if USE_CELL_DATA:
    LATENT_DIM = 100
if USE_MNIST_DATA:
    LATENT_DIM = 20
GAN_LR = 0.0005

#fed variables
FED_DEVICE =  'cpu'
FED_ROUNDS = 20
FED_EPOCHS = 5
FED_MODEL_PATH = '/data/cellface/6_Container/traindata/3types/Separated cells/net'