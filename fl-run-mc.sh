#!/usr/bin/env bash

#Ask the user for number of clients
read -p 'Enter number of Local clients (1 to 32):' nC 


for ((c=1; c<=nC; c++))
do
	konsole --noclose -e python pub-mc.py $c &
done

konsole --noclose -e python sub-mc.py
