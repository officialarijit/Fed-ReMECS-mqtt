import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
import time, queue, sys
import numpy as np
import pandas as pd
import json
from json import JSONEncoder
from federated_utils import *
from Numpy_to_JSON_utils import *

global qGSModel, qGSPerfm
qGSModel = queue.Queue()
qGSPerfm = queue.Queue()


global global_model_result, prev_global_model, current_global_model
global_model_result =[]
prev_global_model = list()

l_rate = 0.05 #Learning rate


def on_connect(client, userdata, flags, rc):
    if rc ==0:
        print("Global Server connected to broker successfylly ")
    else:
        print(f"Failed with code {rc}")

    for i in topic_list:
        val = client.subscribe(i)
        print(val)


def on_message(client, userdata, message):
    if (message.topic == "LocalModel"):
        print("Message received from Local Model")
        qGSModel.put(message)

    if(message.topic == 'ModelPerformance'):
        print('Performance metric received  from Local Models')
        qGSPerfm.put(message)


print('---------------------------------------------------------------')
# mqttBroker = "mqtt.eclipseprojects.io"
mqttBroker = "broker.hivemq.com"
client = mqtt.Client(client_id ="GlobalServer", clean_session=True)
client.on_connect = on_connect
client.connect(mqttBroker,1883)

topic_list =[('LocalModel',0),('ModelPerformance',0)]

client.loop_start()
client.on_message = on_message

#**********************************************************
time.sleep(5) #Wait for connection setup to complete
#**********************************************************

print('---------------------------------------------------------------')


i = 0


while True:
    print('---------STARTED-------------')
    print('Global Server')
    print('Round: ',i)

    time.sleep(5)
    #====================================================
    # Global Model Performance Printing
    #====================================================

    if (i>0): #after first round of model exchange global models performance is calculated
        print('Now Collecting Local Model erformacen Metrics....')
        local_model_performace = list()
        while not qGSPerfm.empty():
            message = qGSPerfm.get()

            if message is None:
                continue

            msg_model_performance = message.payload.decode('utf-8')

            decodedModelPerfromance = list(json.loads(msg_model_performance).values())
            local_model_performace.append(decodedModelPerfromance)

        global_model_performance = np.array(local_model_performace)
        global_performance = np.mean(global_model_performance, axis=0)

        len_local_perfm = len(local_model_performace)
        print('Total Model Performance received:',len_local_perfm)

        if (len_local_perfm != 0):
            global_model_result.append([i+1,i+1,global_performance[1],global_performance[2],global_performance[3],global_performance[4]])
            print('----------------------------------------------------')
            print('Global Model Valence Accuracy:',global_performance[1])
            print('Global Model Valence F1-score:',global_performance[2])
            print('Global Model Aroual Accuracy:',global_performance[3])
            print('Global Model Arousal F1-score:',global_performance[4])
            print('----------------------------------------------------')
        else:
            break #No more data from local model

    #**********************************************************
    time.sleep(50) #to receive model weights
    #**********************************************************

    #=========================================================
    # Local Model Receiving Part
    #=========================================================
    all_local_model_weights = list()

    while not qGSModel.empty():
        message = qGSModel.get()

        if message is None:
            continue

        msg = message.payload.decode('utf-8')

        decodedweights = json2NumpyWeights(msg)

        # Convert object to a list
        local_model_weights = list(decodedweights)
        scaled_weights = scale_model_weights(local_model_weights, 0.1)
        all_local_model_weights.append(scaled_weights)

    print('Total Local Model Received:',len(all_local_model_weights))

    #======================================================

    i +=1 #Next round increment

    #===================================================================
    # Publish the Global Model after Federated Averaging
    #===================================================================
    if i >0:
        #to get the average over all the local model, we simply take the sum of the scaled weights
        averaged_weights = list()
        averaged_weights = sum_scaled_weights(all_local_model_weights)

        global_weights = EagerTensor2Numpy(averaged_weights)

        if( i ==1):
            prev_global_model = global_weights
            encodedGlobalModelWeights = json.dumps(prev_global_model,cls=Numpy2JSONEncoder)
        else:
            global_weights = global_weights_mul_lr(global_weights, l_rate)
            current_global_model = list()
            for i in range(len(global_weights)):
                current_global_model.append( prev_global_model[i] - global_weights[i])

            prev_global_model = current_global_model
            encodedGlobalModelWeights = json.dumps(current_global_model,cls=Numpy2JSONEncoder)


        client.publish("GlobalModel", payload = encodedGlobalModelWeights) #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model to Topic:--> GlobalModel")

        #**********************************************************
        time.sleep(30) #pause it so that the publisher gets the Global model
        #**********************************************************

        #====================================================================


    print('---------------HERE------------------')
    #===================================================================================
    # If No more data from Publisher exit and server closed connection to the broker
    #===================================================================================
    if(i >0 and len(all_local_model_weights)==0): #loop break no message from producer
        break




#Global Model Result Save
folderPath = '/home/gp/Desktop/MER_arin/FL-mqtt/Federated_Results/'
fname_fm = folderPath +'_Global_Model' +'_'+'_results.csv'
column_names = ['Person','Video','Val_Acc', 'Val_F1', 'Aro_Acc','Aro_F1']
global_model_result = pd.DataFrame(global_model_result,columns = column_names)
global_model_result.to_csv(fname_fm)


print("All done, Global Server Closed.")
client.loop_stop()
