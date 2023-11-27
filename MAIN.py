import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
import cv2

# pickle_in = open('CNN_mnist.pkl', 'rb') 
# CNN_mnist = pickle.load(pickle_in) 

#st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])


def upload_image():
    input_data = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
    if input_data is not None:
        file_bytes = np.asarray(bytearray(input_data.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        return opencv_image


def open_model(model_name):
    load_model = open(model_name, 'rb') 
    model = pickle.load(load_model)
    return model


def pred(input_data,model,model_name):
    if model_name == 'CNN_mnist.pkl':
        print("Model Prediction.\n")
        results = model.predict(input_data)
        results = argmax(results,axis = 1)
        results = pd.Series(results,name="Predicted Label")
        print(results)
        # submission = pd.concat([pd.Series(y_test,name = "Actual Label"),results],axis = 1)
        # submission.to_csv("C:/Users/Lenovo/Desktop/deep-learning/DL-ALGORITHMS/CNN/mnist_sample/results/MNIST-CNN1.csv",index=False)

    if model_name == 'CNN_tumor.pkl':

        img=Image.fromarray(input_data)
        img=img.resize((128,128))
        img=np.array(img)
        input_img = np.expand_dims(img, axis=0)
        res = model.predict(input_img)
        if res:
            st.write("Tumor Detected")
        else:
            st.write("No Tumor")



    if model_name == 'DNN_model.pkl':  
        prediction = model.predict([input_data])
        print('Predicted: %s (class=%d)' % (prediction, argmax(prediction)))      


    if model_name == 'LSTM_code.pkl':
        print('add prediction function')

    if model_name == 'RNN_smsspam.pkl':
        preds = (model.predict(input_data) > 0.5).astype("int32")
        print(preds)



# def main(): 
st.title("Deep Learning")     

option = st.selectbox("choose an Task",('Image classification','Text processing'),index=None,placeholder="Select problem method...",)

st.write('You selected:', option)

if option == None:
    pass

elif option == 'Image classification':
    out = st.radio(
        "Select your prediction ",
        key="visibility",
        options=["digit prediction", "Tumor prediction"],)
    
    if out == "digit prediction":
        st.write('Upload digit prediction data')
        model_name = 'CNN_mnist.pkl'
        model = open_model(model_name)
        input_data = upload_image()
        if input_data is not None:
           st.image(input_data, channels="BGR")


        
         

        
    else:
        st.write('Upload tumer data')
        model_name = 'CNN_tumor.pkl'
        model = open_model('CNN_tumor.pkl')
        input_data = upload_image()
        if input_data is not None:
           st.image(input_data, channels="BGR")

        pred(input_data,model,model_name)       
    
else:
    out =  st.radio(
        "Select your prediction ",
        key="visibility",
        options=["IMBD sentimental", "SMS spam prediction"],)

