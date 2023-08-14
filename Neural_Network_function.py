import re
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier, MLPRegressor



#Load the Neural Network model
model_nn = pickle.load(open('Neural_Network/model.p', 'rb'))
count_vect = pickle.load(open('Neural_Network/feature.p', 'rb'))

#Cleansing Function
def text_cleansing(text):
    clean_text = text.lower()
    clean_text = re.sub(r'[^\w\s]', '',clean_text)
    return clean_text

#Neural Netwotk Model Function for text input
def neural_network_model(text): 
    clean_text = count_vect.transform([text_cleansing(text)])

    sentiment = model_nn.predict(clean_text)[0]
    return sentiment

#Create Function for Neural Network file upload 
def neural_network_upload(file_upload):
    
    # Read csv file upload using encoding latin-1
    nn_file = pd.read_csv(file_upload)

    nn_file = pd.DataFrame(nn_file.iloc[:,0])

    # Rename kolom to "text" 
    nn_file.columns = ["text"]
    
    # Apply the Neural Network sentiment prediction to the text
    nn_file['sentiment'] = nn_file.apply(lambda row : neural_network_model(row['text']), axis = 1)
 

    return nn_file






