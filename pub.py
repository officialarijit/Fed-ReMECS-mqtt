#================================================================================================
# Import important libraries
#================================================================================================
import paho.mqtt.client as mqtt
import time, queue, sys, datetime, json, math, scipy, pywt, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from creme import metrics

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
global all_emo
all_emo = []

fm_acc_val = metrics.Accuracy()
fm_f1m_val = metrics.F1()
fm_roc_val = metrics.ROCAUC()
fm_mcc_val = metrics.MCC()
fm_cm_val  = metrics.ConfusionMatrix()

fm_acc_aro = metrics.Accuracy()
fm_f1m_aro = metrics.F1()
fm_roc_aro = metrics.ROCAUC()
fm_mcc_aro = metrics.MCC()
fm_cm_aro  = metrics.ConfusionMatrix()
#=================================================================================================

#=================================================================================================

n = sys.argv[1] #Reading the command line argument passed ['filename.py','passed value/client number']

client_name = 'LocalServer (User)'+n

print(client_name +':>>' +' ' +'Streaming Strated!')

p = int(n) #Person number

#=================================================================================================

#=================================================================================================
# All MQTT ones Here
#=================================================================================================

global qLS
qLS = queue.Queue() #Queue to store the received message in on_message call back

def on_connect(client, userdata, flags, rc): #on connect callback from MQTT
    if rc ==0:
        print("Local Server connected to broker successfully")
    else:
        print(f"Failed with code {rc}")

def on_message(client, userdata, message): #On message callback from MQTT
    print('Global Model Received after FedAvg')
    qLS.put(message)


# mqttBroker = "mqtt.eclipseprojects.io" #Used MQTT Broker
mqttBroker = "broker.hivemq.com"

client = mqtt.Client(client_name) #mqtt Client
client.on_connect = on_connect
client.connect(mqttBroker, 1883) #mqtt broker connect
client.loop_start()

time.sleep(5) #Wait for connection setup to complete


#=================================================================================================

#------------------------------------
# Once file fetched data stored here
#------------------------------------
# grand_eeg = eeg_data(p)
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
classifier = 'FFNN_Feature_Fusion'
init_m = 0
indx = 0
c=0
ccc =0
i =0
videos = 32 #Total Number of Videos
#=================================================================================================


print('Working with -->', classifier)
#=======================================
# MAIN Loop STARTS HERE
#=======================================
for jj in range(0,videos): #Video loop for each participants
    v = jj+1 #Video number
    print('=========================================================================')
    p_v = 'Person:'+ ' ' +str(p)+ ' ' +'Video:'+str(v)
    print(p_v)

    emotion_label =[]


    ##===================================================
    # Data read from files
    ##===================================================
    eda_sig = grand_eda.loc[grand_eda.iloc[:,1] == v]
    resp_sig = grand_resp.loc[grand_resp.iloc[:,1] == v]

    #=================================================
    #emotion labels (valence, arousal) mapping 0-1
    #=================================================
    val = eda_sig.iloc[0,8067]
    aro = eda_sig.iloc[0,8068]


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


    #=========================================
    # Sliding window starts here
    #=========================================
    slider_eda = Slider(bucket_size,overlap_count)
    slider_resp = Slider(bucket_size,overlap_count)

    eda_sig = np.array(eda_sig.iloc[:,range(3,8067)]) #keeping only eda signals
    resp_sig = np.array(resp_sig.iloc[:,range(3,8067)]) #keeping only resp signals

    slider_eda.fit(eda_sig)
    slider_resp.fit(resp_sig)

    while True:
        window_data_eda = slider_eda.slide()
        window_data_resp = slider_resp.slide()


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


        x_FF = np.array([np.concatenate([x_eda,x_resp], axis=None)]) #Fused Feature

        #===================================================
        # Model initialization
        #===================================================
        if init_m == 0:
            print('EDA Feature shape{}:'.format(x_eda.shape))
            print('RESP BELT Feature shape{}:'.format(x_resp.shape))
            print('Fused Feature shape{}:'.format(x_FF.shape))

            #==============================
            # Feature Fused Model
            #==============================
            fm_model = create_model(x_FF, 'Fusion_Model')

            init_m = init_m+1


        #===============================================================
        # Emotion Classification --> Valence and Arousal
        #===============================================================

        if c == 0: #For the first time model will return 0 or None
            tmp_y = [0,0]

            y_pred_FusedModel = [9,9]
            fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

            c=c+1

        else:
            tmp_y = fm_model.predict(x_FF)
            fm_model.fit(x_FF, y_act, epochs = 1, batch_size = 1, verbose=0)

        if slider_eda.reached_end_of_list():
            break


    #===========================================
    # Performance matric update
    #===========================================
    y_pred_FusedModel = np.where(tmp_y[0] > 0.5, 1, 0)

    y_act_aro = y_act[0][1] #actual Arousal
    y_act_val = y_act[0][0] #actual Valence

    y_pred_val = y_pred_FusedModel[0]
    y_pred_aro = y_pred_FusedModel[1]

    fm_acc_val = fm_acc_val.update(y_act_val, y_pred_val)  # update the accuracy metric

    fm_f1m_val = fm_f1m_val.update(y_act_val, y_pred_val) #update f1 measure metric

    fm_roc_val = fm_roc_val.update(y_act_val, y_pred_val)

    fm_mcc_val = fm_mcc_val.update(y_act_val, y_pred_val)

    fm_cm_val = fm_cm_val.update(y_act_val, y_pred_val)


    fm_acc_aro = fm_acc_aro.update(y_act_aro, y_pred_aro)  # update the accuracy metric

    fm_f1m_aro = fm_f1m_aro.update(y_act_aro, y_pred_aro) #update f1 measure metric

    fm_roc_aro = fm_roc_aro.update(y_act_aro, y_pred_aro)

    fm_mcc_aro = fm_mcc_aro.update(y_act_aro, y_pred_aro)

    fm_cm_aro = fm_cm_aro.update(y_act_aro, y_pred_aro)

    print('----------------------------------------------------')
    print('Actual Class:',y_act[0])
    print('Fusion Model predicted:{}'.format(y_pred_FusedModel))


    print('Model Valence accuracy:{}'.format(round(fm_acc_val.get(),4)))
    print('Model Valence f1-score:{}'.format(round(fm_f1m_val.get(),4)))

    print('Model Arousal accuracy:{}'.format(round(fm_acc_aro.get(),4)))
    print('Model Arousal f1-score:{}'.format(round(fm_f1m_aro.get(),4)))
    print('----------------------------------------------------')

    all_emo.append([p,v, fm_acc_val, fm_f1m_val, fm_acc_aro, fm_f1m_aro, y_act[0][0],y_pred_val ,y_act[0][1], y_pred_aro])


    #==========================================================
    # Model weight compress into JSON format
    #==========================================================

    #Message Generation and Encoding into JSON
    model_weights = fm_model.get_weights()
    encodedModelWeights = json.dumps(model_weights,cls=Numpy2JSONEncoder)

    # print(encodedModelWeights)

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
        while not qLS.empty():
            message = qLS.get()

            if message is None:
                continue

            msg = message.payload.decode('utf-8')

            # Deserialization the encoded received JSON data
            global_weights = json2NumpyWeights(msg)

            fm_model.set_weights(global_weights) #Replacing the old model with the newley received model from Global Server

    #===========================================================================
    # All Done Results Save Here
    #===========================================================================
    if (i == videos):
        #if all the videos are done means no more data from User
        #Save all the results into CSV file
        folderPath = '/home/gp/Desktop/MER_arin/FL-mqtt/Federated_Results/'
        fname_fm = folderPath + client_name +'_person_FusionModel'+'_'+'_results.csv'
        column_names = ['Person', 'Video', 'Val_Acc', 'Val_F1', 'Aro_Acc','Aro_F1', 'y_act_val', 'y_act_aro', 'y_pred_aro', 'y_pred_aro']
        all_emo = pd.DataFrame(all_emo,columns = column_names)
        all_emo.to_csv(fname_fm)

        print('All Done! Client Closed')
