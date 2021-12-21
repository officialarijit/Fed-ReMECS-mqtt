import paho.mqtt.client as mqtt
from random import randrange, uniform
import time
import queue



q = queue.Queue()

def on_message(client, userdata, message):
    q.put(message)


mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client("Producer 3")
client.connect(mqttBroker)
model = 0

i =0

while True:
    print('---------------STRATED-----------------')
    print('Local Server 3')
    print('Round: ',i)

    print('Model val before:', model)
    randNumber = str(randrange(5))
    client.publish("TEMPERATURE", randNumber)
    print("Just published:--> " + str(randNumber) + " to Topic TEMPERATURE")
    time.sleep(2)

    client.subscribe("AVGTMP")
    client.loop_start()
    client.on_message = on_message
    if i>0:
        #===============================================================================
        # Publisher as subscriber to receive results after operation at Subscriber end
        #===============================================================================
        # client.on_message = on_message
        while not q.empty():
            message = q.get()

            if message is None:
                continue

            msg = message.payload.decode('utf-8')
            model = msg
            print('Model val after:',model)
    i +=1
