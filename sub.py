import paho.mqtt.client as mqtt
import time, queue, sys
import numpy as np
import json
from json import JSONEncoder
from federated_utils import *
from Numpy_to_JSON_utils import *

global qGS
qGS = queue.Queue()

def on_connect(client, userdata, flags, rc):
    if rc ==0:
        print("Global Server connected to broker successfully")
    else:
        print(f"Failed with code {rc}")


def on_message(client, userdata, message):
    print("Message received from Local Model")
    qGS.put(message)

# mqttBroker = "mqtt.eclipseprojects.io"

mqttBroker = "broker.hivemq.com"
client = mqtt.Client(client_id ="GlobalServer", clean_session=True)
client.on_connect = on_connect
client.connect(mqttBroker, 1883)

client.loop_start()
client.subscribe("LocalModel")
client.on_message = on_message

time.sleep(5) #Wait for connection setup to complete


i = 0


while True:
    all_local_model_weights = list()
    local_clients = list()
    print('---------STARTED-------------')
    print('Global Server')
    print('Round: ',i)

    time.sleep(50)

    while not qGS.empty():
        message = qGS.get()

        if message is None:
            continue

        msg = message.payload.decode('utf-8')

        decodedweights = json2NumpyWeights(msg)

        # Convert object to a list
        local_model_weights = list(decodedweights)
        scaled_weights = scale_model_weights(local_model_weights, 0.1)
        all_local_model_weights.append(scaled_weights)

    print('Total Local Model Received:',len(all_local_model_weights))

    i +=1 #Next round increment

    if i >0:
        #===================================================================
        # Publish into Different topic after performing operation
        #===================================================================
        #to get the average over all the local model, we simply take the sum of the scaled weights
        averaged_weights = list()
        averaged_weights = sum_scaled_weights(all_local_model_weights)
        global_weights = EagerTensor2Numpy(averaged_weights)

        encodedGlobalModelWeights = json.dumps(global_weights,cls=Numpy2JSONEncoder)

        client.publish("GlobalModel", payload = encodedGlobalModelWeights) #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model to Topic:--> GlobalModel")

        #====================================================================

    if(i >0 and len(all_local_model_weights) ==0): #loop break no message from producer
        break

    print('---------------HERE------------------')



print("All done, Global Server Closed.")
client.loop_stop()
