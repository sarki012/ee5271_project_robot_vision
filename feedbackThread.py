# Erik Sarkinen

from pylibftdi import Device
import os
import glob
import time
from bluetooth import *
import socket
import bluetooth
import cv2
import numpy as np
import tty, sys, termios
import threading
import streamThread
import rcThread

feedBack = 0

def task2():
    global feedBack
    myString = 0
    breakFlag = 0
    while True:
        try:  
            ##########Data Relay From dsPIC UART to Bluetooth#########
            adcVals = rcThread.deviceUC1.read(10)
          #  print(len(adcVals))
          #  print(adcVals)
            if len(adcVals) == 10:
                for element in range(0, len(adcVals)):
                    if adcVals[element] == 'x':
                        breakFlag = 1
                        break
                if breakFlag != 1:    
                    myString = adcVals.split(b'c')
                        #  print(myString)
                    feedBack = int(myString[1])
               #     print(feedBack)
                 #   time.sleep(1)
                breakFlag = 0
                #print(adcVals)
             #   print(feedBack)
            
        #    rcThread.client_sock.send(adcVals)
        #   time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nDisconnected")
            raise SystemExit 