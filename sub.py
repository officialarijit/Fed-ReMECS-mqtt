import paho.mqtt.client as mqtt
import time, queue, sys
import numpy as np
import json
from json import JSONEncoder
from federated_utils import *
from Numpy_to_JSON_utils import *

global q
q = queue.Queue()

def on_connect(client, userdata, flags, rc):
    if rc ==0:
        print("Global Server connected to broker successfully")
    else:
        print(f"Failed with code {rc}")


def on_message(client, userdata, message):
    print("Message received from Local Model")
    q.put(message)

mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client(client_id ="GlobalServer", clean_session=True)
client.on_connect = on_connect
client.connect(mqttBroker)

client.loop_start()
client.subscribe("LocalModel")
client.on_message = on_message

time.sleep(5) #Wait for connection setup to complete


i = 0


while True:
    all_local_model_weights = list()
    print('---------STARTED-------------')
    print('Global Server')
    print('Round: ',i)

    time.sleep(50)

    while not q.empty():
        message = q.get()

        if message is None:
            continue

        msg = message.payload.decode('utf-8')

        # Deserialization  of received JS0N data to Numpy Array
        decodedModelWeights = json.loads(msg)
        finalModelWeights = np.asarray(list(decodedModelWeights.values())[0])

        #scale the model weights and add to list
        scaled_weights = scale_model_weights(finalModelWeights, 0.1)
        all_local_model_weights.append(scaled_weights)

    print(len(all_local_model_weights))

    i +=1 #Next round increment

    if i >0:
        #===================================================================
        # Publish into Different topic after performing operation
        #===================================================================
        #to get the average over all the local model, we simply take the sum of the scaled weights
        global_weights = sum_scaled_weights(all_local_model_weights)
        encodedGlobalModelWeights = json.dumps(global_weights,cls=NumpyEncoder)

        client.publish("GlobalModel", payload = encodedGlobalModelWeights) #str(Global_weights), qos=0, retain=False)
        print("Broadcasted Global Model to Topic:--> GlobalModel")

        #====================================================================

    if(i >0 and len(all_local_model_weights) ==0): #loop break no message from producer
        break

    print('---------------HERE------------------')



print("All done, Global Server Closed.")
client.loop_stop()
