import cv2
import numpy as np
import os

class BoWExtractor():

    def __init__ (self,threshold,cluster_num):

        self.image_paths = []
        self.image_keypoints = {}
        self.surf = cv2.xfeatures2d.SURF_create(threshold)
        self.bow_kmeans = cv2.BOWKMeansTrainer(cluster_num)
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.surf, cv2.BFMatcher(cv2.NORM_L2))
        self.empty_histogram = [0.0] * cluster_num
        self.count = 0


    def read_image(self,image_dirs):

        for image_dir in image_dirs:
            image_names = os.listdir(image_dir)
            for image_name in image_names:
                image_path = image_dir + "/" + image_name
                self.image_paths.append(image_path)


    def detector(self):

        for image_path in self.image_paths:
            img = cv2.imread(image_path)
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            key_points,descriptor = self.surf.detectAndCompute(gray,None)

            self.bow_kmeans.add(descriptor)
            self.image_keypoints[image_path] = key_points

            self.count += 1
            if self.count % 10 == 0:
                print (self.count)


    def clustering(self):

        print "start clustering"
        vocabulary = self.bow_kmeans.cluster()
        print (len(vocabulary))
        self.bow_extractor.setVocabulary(vocabulary)

    def save_vocabulary(self, save_path):

        vocabulary = self.bow_extractor.getVocabulary()
        
        if '.npy' in save_path:
            np.save(save_path, vocabulary)
        elif '.npz' in save_path:
            np.savez(save_path, vocabulary)
        else:
            raise NotImplementedError()
    

    def load_vocabulary(self, load_path):

        vocabulary = np.load(load_path)
        self.bow_extractor.setVocabulary(vocabulary)


    def extract(self, image_path):

        img = cv2.imread(image_path)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        keypoints = self.image_keypoints[image_path]
        if keypoints is None or len(keypoints) == 0:
            return self.empty_histogram
        
        return self.bow_extractor.compute(gray, keypoints)[0]

if __name__ == '__main__':
    
    threshold = 500
    cluster_num = 1000
    save_path = "../../weights/bow_vocabulary.npy"
    # load_path = "/home/he/Kit/Masterarbeit/3D_Mapping_ws/src/object_detector/photos/rgb/bow_vocabulary.npy"
    image_dirs = ("rgb/positive",
                "rgb/negative")
    bow = BoWExtractor(threshold,cluster_num)
    bow.read_image(image_dirs)
    bow.detector()
    bow.clustering()
    bow.save_vocabulary(save_path)
    print('end saving')
  
