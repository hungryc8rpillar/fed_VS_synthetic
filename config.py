#paths
RBC_path_train = '/data/cellface/6_Container/traindata/3types/Separated cells/21-10-22_RBC_M1.seg'
#PLT_path_train = '/data/cellface/6_Container/traindata/3types/Separated cells/03-11-22_Ficoll_PLT_M1.seg'
PLT_path_train = '/data/cellface/6_Container/seg_yolo/25-01-23/PLT_Isolation/PLT-Isolation_M1.seg'

RBC_path_test = '/data/cellface/6_Container/traindata/3types/Separated cells/21-10-22_RBC_M2.seg'
#PLT_path_test = '/data/cellface/6_Container/traindata/3types/Separated cells/03-11-22_Ficoll_PLT_M2.seg'
PLT_path_test = '/data/cellface/6_Container/seg_yolo/25-01-23/PLT_Isolation/PLT-Isolation_M5.seg'

CLASS_PATHS_TRAIN = [RBC_path_train,PLT_path_train]
CLASS_PATHS_TEST = [RBC_path_test,PLT_path_test]

CLASS_NAMES = ['RBC','PLT']

#general variables
USE_CELL_DATA = False
USE_MNIST_DATA = True

if USE_CELL_DATA:
    SAMPLE_LIMIT = 2000
if USE_MNIST_DATA:
    SAMPLE_LIMIT = 10000
BATCH_SIZE = 64
BALANCE = True
IMG_SIZE = 96
NUM_PARTITIONS = 2

#classifier variables
CLASSIFIER_EPOCHS = 15 #45
CLASSIFIER_DEVICE = 'cuda'

#privacy variables
DELTA = 1e-3
EPSILON = [1000,0.05]

#gan variables
GAN_BATCH_SIZE = 8
GAN_EPOCHS = 10
GAN_DEVICE = 'cuda'
if USE_CELL_DATA:
    LATENT_DIM = 100
if USE_MNIST_DATA:
    LATENT_DIM = 20

#fed variables
FED_DEVICE =  'cpu'
FED_ROUNDS = 5 #8
FED_EPOCHS = 3 #30
FED_MODEL_PATH = '/data/cellface/6_Container/traindata/3types/Separated cells/net'