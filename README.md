# A Federated Learning Method for Real-time Emotion State Classification from Multi-modal Streaming

Emotional and physical health are strongly connected and should be taken care of simultaneously to ensure completely healthy persons. A person's emotional health can be determined by detecting emotional states from various physiological measurements (EDA, RB, EEG, etc.). Affective Computing has become the field of interest, which uses software and hardware to detect emotional states. In the IoT era, wearable sensor-based real-time multi-modal emotion state classification has become one of the hottest topics. In such setting, a data stream is generated from wearable-sensor devices, data accessibility is restricted to those devices only and usually a high data generation rate should be processed to achieve real-time emotion state responses. Additionally, protecting the users' data privacy makes the processing of such data even more challenging. Traditional classifiers have limitations to achieve high accuracy of emotional state detection under demanding requirements of decentralized data and protecting users' privacy of sensitive information as such classifiers need to see all data. Here comes the federated learning, whose main idea is to create a global classifier without accessing the users' local data. Therefore, we have developed a federated learning framework for real-time emotion state classification using multi-modal physiological data streams from wearable sensors, called Fed-ReMECS. In our framework we have been able to address all the above mentioned demanding requirements. The experimental study is conducted using the popularly used multi-modal benchmark DEAP dataset for emotion classification. The results show the effectiveness of our developed approach in terms of accuracy, efficiency, scalability and users' data privacy protection.


## Installation 
- Programming language
  - `Python 3.6`

- Operating system
  - `Ubuntu 18.04 (64 bit)` 

- Required packages
  - `Keras` 
  - `Tensorflow` &#8592; for developing the `neural network`.
  - `Scikit-Learn` &#8592; for model's performance matrics. 
  - `paho-mqtt` &#8592; for `MQTT` protocol implementations. 
  - `Konsole - KDE's Terminal Emulator` &#8592; Terminal emulator.
  -  `Mosquitto MQTT Broker`
  
- Installation steps:
  - Step 1: Install `Anaconda`. 
  - Step 1: Create a `virtual environment` in Anaconnda.
  - Step 2: Run `pip install -r requirements.txt`
  - Step 3: Install `mosquitto broker`
      - `sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa`
      - `sudo apt-get update`
      - `sudo apt-get install mosquitto`
      - `sudo apt-get install mosquitto-clients`
      - `sudo apt clean`
  - Step 4: Open `terminal`, and `activate environment`.
  - Step 5: Run `bash trigger-clients.sh`
  - Step 6: Enter the number of `clients` you want. 
  - Step 7: Enjoy the visuals :wink:.


# NOTE*: Please feel free to use the code by giving proper citation and star to this repository.
