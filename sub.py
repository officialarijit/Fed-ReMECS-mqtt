import paho.mqtt.client as mqtt
import time
import queue

global q
q = queue.Queue()

def on_message(client, userdata, message):
    q.put(message)

    # print("Received message: ", str(message.payload.decode("utf-8")))

mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client(client_id ="Smartphone", clean_session=True)
client.connect(mqttBroker)

client.loop_start()
client.subscribe("TEMPERATURE")



msg_counter = 0

all_message = []
i = 0


while True:
    print('---------STARTED-------------')
    print('Global Server')
    print('Round: ',i)

    time.sleep(2)
    client.on_message = on_message

    while not q.empty():
        message = q.get()

        if message is None:
            continue

        msg = message.payload.decode('utf-8')
        all_message.append(float(msg))

    print(all_message)

    if i >0:
        #===================================================================
        # Publish into Different topic after performing operation
        #===================================================================
        sum_val = sum(all_message)/len(all_message)
        client.publish("AVGTMP", sum_val)
        print("Just published:--> " + str(sum_val) + " to Topic AVGTMP")

        #====================================================================

    if(i >0 and len(all_message) ==0): #loop break no message from producer
        break

    i +=1 #Next round increment

    all_message = []
    print('---------------HERE------------------')



print("All Soen, Global Server Closed.")
client.loop_stop()

# i = 0
# while True:
#     print('---------STARTED-------------')
#     i +=1
#     print('Round: ',i)
#     time.sleep(1)
#     message = all_messages.pop()
#     print('---------------HERE------------------')
