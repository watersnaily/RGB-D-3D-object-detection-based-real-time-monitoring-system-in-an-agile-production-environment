import cv2
import numpy as np
import os
from bow_extractor import BoWExtractor



def svm_data(pos_path,neg_path,load_path,bow_threshold,bow_cluster_num):
    bow = BoWExtractor(bow_threshold,bow_cluster_num)
    bow.read_image((pos_path,neg_path))
    bow.detector()
    bow.load_vocabulary(load_path)

    pos_image = os.listdir(pos_path)
    neg_image = os.listdir(neg_path)
    feature_data = []
    label_data = []
    for image_name in pos_image:
        image_path = pos_path + "/" + image_name
        feature = bow.extract(image_path)
        feature_data.append(feature)
        label_data.append([1])
    for image_name in neg_image:
        image_path = neg_path + "/" + image_name
        feature = bow.extract(image_path)
        feature_data.append(feature)
        label_data.append([0])

    feature_data = np.float32(feature_data)
    label_data = np.array(label_data)

    rand = np.random.RandomState(0)
    shuffle = rand.permutation(len(label_data))
    feature_data = feature_data[shuffle]
    label_data = label_data[shuffle] 

    return feature_data,label_data

class svm_classifier():

    def __init__ (self):
        self.svm = cv2.ml.SVM_create()

    def setup(self):
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_RBF)
        self.svm.setGamma(5.383)
        self.svm.setC(2.67) 

    def train_and_save(self,feature,label,save_path):
        print ('start training')
        self.svm.trainAuto(feature, cv2.ml.ROW_SAMPLE, label,
            kFold = 10,
            balanced = True ) 
        self.svm.save(save_path)

    def validation(self,feature,label):
        feature = np.float32(feature)
        feature = feature.reshape(1,-1)
        _,label_predict = self.svm.predict(feature)
        return label_predict[0,0] == label[0]
    
    # def predict(self,load_path,data_path):
    #     self.svm_trained = cv2.ml.SVM_load(load_path)
    #     feature = 
    #     _,y = self.svm_trained.predict


if __name__ == '__main__':
    pos_path = "rgb/positive"
    neg_path = "rgb/negative"
    bow_load_path = "../../weights/bow_vocabulary.npy"
    svm_save_path = "../../weights/svm_classifier.mat"
    bow_threshold = 500
    bow_cluster_num = 1000
    svm = svm_classifier()

    feature,label = svm_data(pos_path,neg_path,bow_load_path,bow_threshold,bow_cluster_num)
    total_num = len(label)
    feature_train = feature[:total_num*3//4,:]
    label_train = label[:total_num*3//4,:]
    feature_val = feature[total_num*3//4:,:]
    label_val = label[total_num*3//4:,:]
    svm.setup()
    svm.train_and_save(feature_train,label_train,svm_save_path)
    print ('Done')

    print ('start validation')
    c = 0
    for i in range(len(label_val)):
        c += svm.validation(feature_val[i],label_val[i])
    print (c)
    print (len(label_val))
    print(float(c)/len(label_val))
