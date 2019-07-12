from flask import Flask
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential , load_model
from keras.layers import Dense , Dropout , Activation , Flatten
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
import math

app = Flask(__name__)

@app.route("/pred/<arg>")
def pred(arg):
    
    inputs = arg.split("-")
    
    ifc = np.array([inputs[0]]).astype(int)
    iv = np.array([inputs[1]]).astype(float)
    rate = np.array([inputs[2]]).astype(int)
    olt = np.array([inputs[3]]).astype(int)
    ot = np.array([inputs[4]]).astype(int)
    os = np.array([inputs[5]]).astype(int)
    yo = np.array([inputs[6]]).astype(int)
    itc = np.array([inputs[7]]).astype(int)
    it = np.array([inputs[8]]).astype(int)
    
    if ifc == 0:
        fat = [1,0]
    elif ifc == 1:
        fat = [0,1]
        
    if ot == 0:
        outtype = [1,0,0,0]
    elif ot == 1:
        outtype = [0,1,0,0]
    elif ot == 2:
        outtype = [0,0,1,0]
    elif ot == 3:
        outtype = [0,0,0,1]
        
    if olt == 0:
        locality = [1,0,0]
    elif olt == 1:
        locality = [0,1,0]
    elif olt == 2:
        locality = [0,0,1]
        
    if itc == 0:
        combined = [1,0,0]
    elif itc == 1:
        combined = [0,1,0]
    elif itc == 2:
        combined = [0,0,1]
        
    if os == 0:
        outsize = [1,0,0]
    elif os == 1:
        outsize = [0,1,0]
    elif os == 2:
        outsize = [0,0,1]
        
    if it == 0:
        type = [1,0,0,0]
    elif ot == 1:
        type = [0,1,0,0]
    elif ot == 2:
        type = [0,0,1,0]
    elif ot == 3:
        type = [0,0,0,1]
    
    keras.backend.clear_session()
    
    print(rate)
    
    model= load_model("Grocery.h5")
    c = model.predict(np.concatenate((iv,
                                rate ,
                                yo,
                                fat,
                                outtype,
                                locality,
                                combined,
                                outsize) , axis = 0).reshape(1,18,1))[0][0]
    print(c)
    
    keras.backend.clear_session()
    
    pred = str(int(math.ceil(c / 100.0)) * 100)

    return pred

if __name__ == "__main__":
    app.run()
