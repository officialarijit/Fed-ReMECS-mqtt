#================================================================================================
# Import important libraries
#================================================================================================
import paho.mqtt.client as mqtt
import time, queue, sys, datetime, json, math, scipy, pywt, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from json import JSONEncoder
from statistics import mode
from scipy import stats
from sklearn import preprocessing
from collections import defaultdict, Counter
from scipy.special import expit
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from window_slider import Slider


from feature_extraction_utils import *
from data_reading_utils import *
from model_creation import *
from Numpy_to_JSON_utils import *


#=================================================================================================
global eeg_emotion, eda_emotion, resp_emotion, mer_emotion
eeg_emotion = []
eda_emotion = []
resp_emotion = []
mer_emotion = []
#=================================================================================================

#=================================================================================================

n = sys.argv[1] #Reading the command line argument passed ['filename.py','passed value/client number']

client_name = 'LocalServer (User)'+n

print(client_name +':>>' + 'Streaming Strated!')

p = int(n) #Person number

#=================================================================================================

#=================================================================================================
# All MQTT ones Here
#=================================================================================================

global q
q = queue.Queue() #Queue to store the received message in on_message call back

def on_connect(client, userdata, flags, rc): #on connect callback from MQTT
    if rc ==0:
        print("Local Server connected to broker successfully")
    else:
        print(f"Failed with code {rc}")

def on_message(client, userdata, message): #On message callback from MQTT
    q.put(message)


mqttBroker = "mqtt.eclipseprojects.io" #Used MQTT Broker
client = mqtt.Client(client_name) #mqtt Client
client.on_connect = on_connect
client.connect(mqttBroker) #mqtt broker connect
client.loop_start()

time.sleep(5) #Wait for connection setup to complete


#=================================================================================================

#------------------------------------
# Once file fetched data stored here
#------------------------------------
grand_eeg = eeg_data(p)
grand_eda = eda_data(p)
grand_resp = resp_data(p)

#=================================================================================================


#=================================================================================================
# Sliding Window
#=================================================================================================

segment_in_sec = 10 #in sec
bucket_size = int((8064/60)*segment_in_sec)  #8064 is for 60 sec record
overlap_count = 0

#================================================
# Model name and other loop control parameters
#================================================
classifier = 'FFNN'
init_m = 0
indx = 0
c=0
ccc =0
i =0
videos = 32 #Total Number of Videos
#=================================================================================================


for jj in range(0,videos): #Video loop for each participants
            v = jj+1 #Video number
            print('=========================================================================')
            p_v = 'Person:'+ ' ' +str(p)+ ' ' +'Video:'+str(v)
            print(p_v)

            emotion_label =[]


            ##===================================================
            # Data read from files
            ##===================================================
            eeg_sig = grand_eeg.loc[grand_eeg.iloc[:,1] == v]
            eda_sig = grand_eda.loc[grand_eda.iloc[:,1] == v]
            resp_sig = grand_resp.loc[grand_resp.iloc[:,1] == v]

            #=================================================
            #emotion labels (valence, arousal) mapping 0-1
            #=================================================
            val = eeg_sig.iloc[0,8067]
            aro = eeg_sig.iloc[0,8068]


            #==================================================================
            # Useful For classification
            #==================================================================

            #valence emotion maping 0-> low valence and 1-> high valence

            if (val >5):
                vl = 1 #high valence
            else:
                vl = 0 #low valence

            #arousal emotion maping 0-> low arousal and 1-> high high arousal
            if (aro >5):
                al = 1 #high arousal
            else:
                al = 0 #low arousal

            y_act = np.array([[vl,al]])

            #==================================================================

            #==========================================================
            # Predicted Valence and Arousal labels list initialization
            #==========================================================
            eeg_val_prdt=[]
            eeg_aro_prdt =[]

            eda_val_prdt=[]
            eda_aro_prdt =[]

            resp_val_prdt=[]
            resp_aro_prdt =[]


            #=========================================
            # Sliding window starts here
            #=========================================
            slider_eeg = Slider(bucket_size,overlap_count)
            slider_eda = Slider(bucket_size,overlap_count)
            slider_resp = Slider(bucket_size,overlap_count)

            eeg_sig = np.array(eeg_sig.iloc[range(0,32),range(3,8067)]) #keeping only eeg signals
            eda_sig = np.array(eda_sig.iloc[:,range(3,8067)]) #keeping only eda signals
            resp_sig = np.array(resp_sig.iloc[:,range(3,8067)]) #keeping only resp signals

            slider_eeg.fit(eeg_sig)
            slider_eda.fit(eda_sig)
            slider_resp.fit(resp_sig)

            while True:
                window_data_eeg = slider_eeg.slide()
                window_data_eda = slider_eda.slide()
                window_data_resp = slider_resp.slide()

                #=================================================
                # Feature extraction from EEG
                #=================================================
                features_eeg = eeg_features(window_data_eeg)
                eeg = np.array([features_eeg])  #EEG raw feature vector
                x_eeg = np.array(preprocessing.normalize(eeg)) # EEG normalized features [0,1]

                #=================================================
                # Feature extraction from EDA
                #=================================================
                eda_features = extract_eda_features(np.array(window_data_eda))
                eda = np.array([eda_features]) #EDA raw feature vector
                x_eda = np.array(preprocessing.normalize(eda)) #EDA normalized features

                #=================================================
                # Feature extraction from Resp belt
                #=================================================

                resp_features = extract_resp_belt_features(np.array(window_data_resp))
                resp = np.array([resp_features]) #RESP BELT raw feature vector
                x_resp = np.array(preprocessing.normalize(resp)) #RESP BELT normalized features


                #===================================================
                # Model initialization
                #===================================================
                if init_m == 0:
                    print('------------------------------------------------')
                    print('EEG Feature shape{}:'.format(x_eeg.shape))
                    print('EDA Feature shape{}:'.format(x_eda.shape))
                    print('RESP BELT Feature shape{}:'.format(x_resp.shape))
                    print('------------------------------------------------')

                    #========================
                    # For EEG data model
                    #========================
                    eeg_model = create_model(x_eeg)

                    #========================
                    # For EDA data model
                    #========================
                    eda_model = create_model(x_eda)

                    #==============================
                    # For Resp Belt data Model
                    #==============================
                    resp_model = create_model(x_resp)

                    init_m = init_m+1


                #===============================================================
                # Emotion Classification --> Valence and Arousal
                #===============================================================

                if c == 0: #For the first time model will return 0 or None
                    tmp_eeg = [0,0]
                    tmp_eda = [0,0]
                    tmp_resp = [0,0]


                    y_pred_eeg = [2,2]
                    y_pred_eda = [2,2]
                    y_pred_resp = [2,2]

                    eeg_model.fit(x_eeg , y_act, epochs = 1, batch_size = 1, verbose=0)
                    eda_model.fit(x_eda, y_act, epochs = 1, batch_size = 1, verbose=0)
                    resp_model.fit(x_resp, y_act, epochs = 1, batch_size = 1, verbose=0)

                    c=c+1

                else:

                    tmp_eeg = eeg_model.predict(x_eeg)
                    tmp_eda = eda_model.predict(x_eda)
                    tmp_resp = resp_model.predict(x_resp)

                    y_pred_eeg = np.where(tmp_eeg[0] > 0.5, 1, 0)
                    y_pred_eda = np.where(tmp_eda[0] > 0.5, 1, 0)
                    y_pred_resp = np.where(tmp_resp[0] > 0.5, 1, 0)

                    eeg_model.fit(x_eeg, y_act, epochs = 1, batch_size = 1, verbose=0)
                    eda_model.fit(x_eda, y_act, epochs = 1, batch_size = 1, verbose=0)
                    resp_model.fit(x_resp, y_act, epochs = 1, batch_size = 1, verbose=0)


                #===========================================
                # Performance matric update
                #===========================================

                eeg_acc = accuracy_score(y_act[0], y_pred_eeg)  # update the accuracy metric

                eeg_f1m = f1_score(y_act[0], y_pred_eeg, average=None) #update f1 measure metric


                # classification EDA

                eda_acc = accuracy_score(y_act[0], y_pred_eda)  # update the accuracy metric

                eda_f1m = f1_score(y_act[0], y_pred_eda, average=None) #update f1 measure metric



                #classification Resp Belt

                resp_acc = accuracy_score(y_act[0], y_pred_resp)  # update the accuracy metric

                resp_f1m = f1_score(y_act[0], y_pred_resp, average=None) #update f1 measure metric



                if slider_eeg.reached_end_of_list():
                    break



            #==========================================================
            # Model Performance Showing
            #==========================================================
            print('------------------------------------------------')
            print('EEG models accuracy:',eeg_acc)
            print('EEG models f1-score:',np.mean(eeg_f1m))
            print('EDA models accuracy:',eda_acc)
            print('EDA models f1-score:',np.mean(eda_f1m))
            print('RESP BELT models accuracy:',resp_acc)
            print('RESP BELT models f1-score:',np.mean(resp_f1m))
            print('------------------------------------------------')

            #==========================================================
            # Model weight compress into JSON format
            #==========================================================

            #Message Generation and Encoding into JSON
            # eeg_model_weights = np.asarray(eeg_model.get_weights())
            # print(type(eeg_model_weights))

            eeg_model_weights = np.ones((100,200))
            Model_Weights_numpy = {client_name: eeg_model_weights}
            encodedModelWeights = json.dumps(Model_Weights_numpy,cls=NumpyEncoder)

            print(type(encodedModelWeights))

            #==========================================================
            # Broadcast (Publish) Local model weights to the mqttBroker
            #==========================================================

            client.publish("LocalModel", payload = encodedModelWeights)

            print("Local Model Broadcasted for "+ p_v +" to Topic:-> LocalModel")
            time.sleep(60) #put the loca server in sleep for 60 sec

            i +=1 #incrementing this will het the model sent by Global Server

            if i>0: #Receive Global model from the Subscriber end
                #===============================================================================
                # Publisher as subscriber to receive results after operation at Subscriber end
                #===============================================================================
                client.subscribe("GlobalModel")
                client.on_message = on_message
                while not q.empty():
                    message = q.get()

                    if message is None:
                        continue

                    msg = message.payload.decode('utf-8')

                    # Deserialization the encoded received JSON data
                    global_weights = json.loads(msg)
                    finalNumpyModelWeights = np.asarray(list(decodedModelWeights.values())[0])

                    print('Received FedAvg Model From Global Server')

                    # eeg_model.set_weights(global_weights) #Replacing the old model with the newley received model from Global Server
