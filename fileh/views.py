from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2

import tensorflow as tf
from numpy.random import seed
seed(101)

tf.random.set_seed(101)

#import keras
#from keras import backend as K

import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

from PIL import Image
from skimage import transform

def Canny_detector(img, weak_th = None, strong_th = None):
    
      
    # conversion of image to grayscale
    cv2.imwrite('static/testimages1/img1.jpg',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('static/testimages1/grimg1.jpg',img)
       
    # Noise reduction step
    img = cv2.GaussianBlur(img, (7, 7), 1.4)
    cv2.imwrite('static/testimages1/blrimg1.jpg',img)   
    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
       
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    height, width = img.shape
       
    for i_x in range(width):
        for i_y in range(height):
               
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
               
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0
   
    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)              
    ids = np.zeros_like(img)
       
    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):
              
            grad_mag = mag[i_y, i_x]
              
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2
       
    cv2.imwrite('static/testimages1/img2.jpg',mag)
    return mag

def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    
    return np_image
# Create your views here.
def load2(filename):
    
    np_image = Image.open(filename)
    np_image = np.array(np_image, dtype=np.uint8)
    np_img=Canny_detector(np_image)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def upload(request):
    res="Upload an image"
    res2="Upload an image"
    res3="Upload an image"
    
    if request.method == 'POST':
        uploaded_file=request.FILES['document']
        print(uploaded_file.name)
        uploaded_file.name="img1.jpg"
        print(uploaded_file.size)

        fs=FileSystemStorage()
        fs.delete("img1.jpg")
        fs.delete("static/testimages/img1.jpg")
        fs.delete("static/testimages/img2.jpg")
        fs.save(uploaded_file.name,uploaded_file)
        

        worddict={'0':'Bowens disease (akiec)','1':'basal cell carcinoma (bcc)','2':'benign keratosis-like lesions','3':'dermatofibroma (df)','4':'melanoma (mel)', '5':'melanocytic nevi (nv)', '6':'vascular lesions'}



        mobile2 = tensorflow.keras.applications.mobilenet.MobileNet()


        # CREATE THE MODEL ARCHITECTURE

        # Exclude the last 5 layers of the above model.
        # This will include all layers up to and including global_average_pooling2d_1
        x = mobile2.layers[-6].output

        # Create a new dense layer for predictions
        # 7 corresponds to the number of classes
        x = Dropout(0.25)(x)
        predictions2 = Dense(7, activation='softmax')(x)

        # inputs=mobile.input selects the input layer, outputs=predictions refers to the
        # dense layer we created above.

        model2 = Model(inputs=mobile2.input, outputs=predictions2)

        model2.load_weights('F:\main files\input\model2.h5')





        mobile = tensorflow.keras.applications.mobilenet.MobileNet()    

        image2 = load2('C:/Users/Siddharth Raj dash/projects/siddharth/mysite/static/img1.jpg')

        predictions2=model2.predict(image2)
        print(predictions2[0])
        print(predictions2.argmax(axis=1)[0])

        res2=worddict[str(predictions2.argmax(axis=1)[0])]

        x = mobile.layers[-6].output

        x = Dropout(0.25)(x)
        predictions = Dense(7, activation='softmax')(x)

        model = Model(inputs=mobile.input, outputs=predictions)
        model.load_weights('F:\main files\input\model.h5')

        
        image = load('C:/Users/Siddharth Raj dash/projects/siddharth/mysite/static/img1.jpg')
        predictions=model.predict(image)
        print(predictions[0])
        print(predictions.argmax(axis=1)[0]) 
        l=[0,1,2,3,4,5,6]
        h=predictions[0]
        for i in range(7):
            for j in range(7):
                if h[i]>h[j]:
                    h[i],h[j]=h[j],h[i]
                    l[i],l[j]=l[j],l[i]
        print(l)
        print(h)
        
        res=worddict[str(l[0])]
        res2=worddict[str(l[1])]
        res3=worddict[str(l[2])]
    
    return render(request,'upload.html',{'result':res,'result2':res2,'result3':res3})