from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
import warnings
warnings.filterwarnings('ignore')
main = tkinter.Tk()
main.title("Tomato Image Classificaion")
main.geometry("1300x1200")
reverse_mapping = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
                   'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                   'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def upload():
        global filename
        filename = askopenfilename()
        text.insert(END,"Image file is uploaded"+str(filename)+"\n")
	

def modelLoad():
        global model
        model = load_model('plant_disease_final.h5')
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
        text.insert(END,"Pre-trained Model is uploaded.\n")
        print(model.summary)

def predict():
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        image = np.reshape(image,[1,224,224,3])
        #plt.imshow(image)
        #plt.show()
        disease = model.predict_classes(image)
        prediction = disease[0]
        print(prediction)
        prediction_name = reverse_mapping[prediction]
        text.insert(END,"Predicted Class:"+str(prediction_name)+"\n")


font = ('times', 16, 'bold')
title = Label(main, text='Tomato Image Detection and Classification')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')

cvect = Button(main, text="Model Load", command=modelLoad)
cvect.place(x=700,y=100)
cvect.config(font=font1)

download = Button(main, text="Upload Image", command=upload)
download.place(x=700,y=150)
download.config(font=font1)

tfvect = Button(main, text="Prediction", command=predict)
tfvect.place(x=700,y=200)
tfvect.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='pale turquoise')
main.mainloop()
