# Erik Sarkinen

from re import X
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
import robotMain
import rcThread
import feedbackThread
from threading import Event
import signal
import queue

# (x, y) are the center of the cup
# { = left, } = right
# + = up, - = down

def task3():
   # x= 0
    startRight = 0
    endRight = 0
    movedRight = 0
    startLeft = 0
    endLeft = 0
    movedLeft = 0
    direction = 0
    currentTime = 0
    seconds = 0
    secondsStart = 0

    print("Thread 3 started. Waiting for the signal....")
    while True:
        try:
            robotMain.thread_switch_event.wait()
            print("Welcome to Auto Camera Mode.")
            robotMain.go_event.wait()
            print("Go*****************************************!!")

            ##############Left to Right##############################
            if streamThread.x > 497:
                startRight = time.time()
                while streamThread.x > 497:
                    rcThread.deviceUC1.write('}')   #Right
                    time.sleep(1)
                endRight = time.time()
                movedRight = endRight - startRight
                direction = 0
            elif streamThread.x < 497:
                startLeft = time.time()
                while streamThread.x < 497:
                    rcThread.deviceUC1.write('{')   #Left
                    time.sleep(1)
                endLeft = time.time()
                movedLeft = endLeft - startLeft
                direction = 1
            for j in range (0, 10):
              rcThread.deviceUC1.write('q')   #Stop (Break)
            print("Left-To-Right Goal Achieved!")

            #############Move out##################################
            while streamThread.y < 500:
                rcThread.deviceUC2.write('W')   #'W' for out
               # time.sleep(1)
            for j in range (0, 10):
              rcThread.deviceUC2.write('q')   #Stop (Break)
            print("Goal Achieved!")

            #############Squeeze claw until feedback is > 6750
            while feedbackThread.feedBack > 400:
                rcThread.deviceUC1.write('c')   #Claw closed
            for j in range (0, 10):
                rcThread.deviceUC1.write('%')   #Stop claw
            print("Claw Squeezed!")

            ##########Lift the boom up#####################
            for k in range (0, 10000):
                rcThread.deviceUC2.write('u')   #Boom up
            for j in range (0, 10):
                rcThread.deviceUC2.write('@')   #Stop boom
            print("Boom Up!")
            
            #if direction == 0:
            
            ###########Rotate Left####################
            secondsStart = time.time()
            while seconds - secondsStart < 5:
                rcThread.deviceUC1.write('{')   #Left
                seconds = time.time()
            for j in range (0, 10):
              rcThread.deviceUC1.write('q')   #Stop (Break)
            print("Left Goal Achieved!")
 #           secondsStart = time.time()
  #          if direction == 0:
   #         while (seconds - secondsStart) + movedRight < 10:
    #            rcThread.deviceUC1.write('{')   #Left
     #           seconds = time.time()
      #      elif direction == 1:
       #       while (seconds - secondsStart) - movedLeft < 10:
        #        rcThread.deviceUC1.write('{')   #Left
         #       seconds = time.time()


            #############Open claw until feedback is > 600
            while feedbackThread.feedBack < 600:
                rcThread.deviceUC1.write('n')   #Claw open
            for j in range (0, 10):
                rcThread.deviceUC1.write('%')   #Stop claw
            print("Claw Open!")

        #    for m in range(0, 100): 
         #     rcThread.client_sock.send('K')           #Send 'K' for clear to clear the go flag. We don't want the Android to keep sending 'go'
            robotMain.go_event.clear()        #Stop auto mode. This thread will wait until go button is pressed again.

        except KeyboardInterrupt:
            print("\nDisconnected")
            raise SystemExit 