# Fed-ReMECS-mqtt
A Federated Learning Method for Real-time Emotion State Classification from Multi-modal Streaming

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
