import paho.mqtt.client as mqtt
from random import randrange, uniform
import time, queue, sys
import numpy as np
import json
from json import JSONEncoder



n = sys.argv[1] #Reading the command line argument passed ['filename.py','passed value/client number']

client_name = 'LocalServer'+n


q = queue.Queue()

def on_message(client, userdata, message):
    q.put(message)


mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client(client_name)
client.connect(mqttBroker)

#==========================================================
# Numpy array to JSON ENcoding
#==========================================================

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
#==========================================================

model = 0
i =0

# time.sleep()

while True:
    print('---------------STRATED-----------------')
    print(client_name)
    print('Round: ',i)

    # print('Model val before:', model)
    numpyArrayOne = np.random.rand(4,9)


    numpyData = {client_name: numpyArrayOne}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    print(encodedNumpyData)

    client.publish("LocalModelBroadcast", encodedNumpyData)
    print("Local Model Broad Casted of "+" " +"Round" + " " +str(i) +" to Topic:-> LocalModelBroadcast")
    time.sleep(3)

    # client.subscribe("GlobalModelBroadcast")
    # client.loop_start()
    # client.on_message = on_message

    if i>0:
        #===============================================================================
        # Publisher as subscriber to receive results after operation at Subscriber end
        #===============================================================================
        client.subscribe("GlobalModelBroadcast")
        client.loop_start()
        client.on_message = on_message
        while not q.empty():
            message = q.get()

            if message is None:
                continue

            msg = message.payload.decode('utf-8')

            decodedArray = json.loads(msg)
            finalNumpyArray = np.asarray(decodedArray)
            print('Received FedAvg Model From Global Server')
            print(finalNumpyArray)
    i +=1
