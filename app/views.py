from flask import render_template, request
import os
import cv2
from app.face_recognition import faceRecognitionPipeline
import matplotlib.image as matimg

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename

        # save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)

        # get predictions
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        print('ML model predicted successfully')

        # generate the report 
        report = []
        for i, obj in enumerate(predictions):
            gray_scale = obj['roi'] # grayscale image
            eigen_image = obj['eigen_img'].reshape(100,100) # grayscale image
            gender_name = obj['prediction_name'] # gender name
            score = round(obj['score'],2) # probability score

            # save grayscale and eigen image in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eigen_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_scale,cmap='gray')
            matimg.imsave(f'./static/predict/{eigen_image_name}',eigen_image,cmap='gray')

            # save report
            report.append([gray_image_name,eigen_image_name,gender_name,score])

        return render_template('gender.html',fileupload=True,report=report) # POST request

    return render_template('gender.html') # GET request