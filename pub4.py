import paho.mqtt.client as mqtt
from random import randrange, uniform
import time
import queue


q = queue.Queue()

def on_message(client, userdata, message):
    q.put(message)


mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client("Producer 4")
client.connect(mqttBroker)
model = 0

i=0 #Counting Rounds 


while True:
    print('---------------STRATED-----------------')
    print('Local Server 4')
    print('Round: ',i)

    print('Model val before:', model)
    randNumber = str(uniform(10.0, 11.0))
    client.publish("TEMPERATURE", randNumber)
    print("Just published:--> " + str(randNumber) + " to Topic TEMPERATURE")
    time.sleep(2)

    client.subscribe("AVGTMP")
    client.loop_start()

    client.on_message = on_message
    # client.on_message = on_message
    if i>0:
        #===============================================================================
        # Publisher as subscriber to receive results after operation at Subscriber end
        #===============================================================================

        while not q.empty():
            message = q.get()

            if message is None:
                continue

            msg = message.payload.decode('utf-8')
            model = msg
            print('Model val after:',model)
    i +=1
