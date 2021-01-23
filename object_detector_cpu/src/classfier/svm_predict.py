import cv2
import numpy as np
import os

class svm_predictor():

    def __init__ (self,svm_load_path,bow_load_path,bow_threshold):
        self.svm = cv2.ml.SVM_load(svm_load_path)    
        self.surf = cv2.xfeatures2d.SURF_create(bow_threshold)
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.surf, cv2.BFMatcher(cv2.NORM_L2))
        self.load_bow(bow_load_path)

    def load_bow(self,load_path):
        vocabulary = np.load(load_path)
        self.bow_extractor.setVocabulary(vocabulary)

    def predict(self,image_path):
        img = cv2.imread(image_path)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        key_points = self.surf.detect(gray,None)
        feature = self.bow_extractor.compute(gray, key_points)[0]
        feature = np.float32(feature)
        feature = feature.reshape(1,-1)
        _,label = self.svm.predict(feature)

        return label

if __name__ == '__main__':

    bow_load_path = "../weights/bow_vocabulary.npy"
    svm_load_path = "../weights/svm_classifier.mat"
    bow_threshold = 500
    predictor = svm_predictor(svm_load_path,bow_load_path,bow_threshold)
    while True:
        image_name = raw_input("image_name:")
        image_path = "../../photos/capture_corlor"+image_name+".png"
        label = predictor.predict(image_path)
        print (label)
        print ('the label is %d'%label)

