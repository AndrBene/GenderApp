import numpy as np
import sklearn
import pickle
import cv2

# load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # cascade classifier
model_svm = pickle.load(open('./model/model_svm.pickle',mode='rb')) # ML model (SVM)
pca_models  = pickle.load(open('./model/pca_dict.pickle',mode='rb')) # pca dictionary

model_pca = pca_models['pca'] # PCA model
mean_face_arr = pca_models['mean_face'] # mean face


def faceRecognitionPipeline(filename, path=True):
    if path:
        # step-01: read image
        img = cv2.imread(filename) #BGR
    else:
        img = filename
    
    # step-02: convert into gray scale
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # step-03: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(img_gray,1.5,3)
    
    predictions = []
    
    for x,y,w,h in faces:
        roi = img_gray[y:y+h,x:x+w]
        #plt.imshow(roi,cmap='gray')
        #plt.show()
    
        # step-04: normalization (0-1)
        roi = roi / 255.0
    
        # step-05: resize images (100,100)
        if roi.shape[1] >= 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        
        # step-06: Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1,10000)
    
        # step-07: subtract with mean face
        roi_mean = roi_reshape - mean_face_arr
        
        # step-08: get eigen image (apply roi_mean to pca)
        eigen_img = model_pca.transform(roi_mean)
        
        # step-09 Eigen Image for Visualization
        eigen_img_inv = model_pca.inverse_transform(eigen_img)
    
        # step-10: pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_img)
        prob_score = model_svm.predict_proba(eigen_img)
        prob_score_max = prob_score.max()
        
        # step-11: generate report
        text = '%s : %d'%(results[0],prob_score_max*100)
        print(text)
        
        # defining color based on results
        if results[0] == 'male':
            color = (255,255,0)
        else:
            color = (255,0,255)
    
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
    
        output = {
            'roi': roi,
            'eigen_img': eigen_img_inv,
            'prediction_name': results[0],
            'score': prob_score_max
        }
    
        predictions.append(output)

    return img, predictions