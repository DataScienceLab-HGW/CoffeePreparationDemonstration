#!/usr/bin/python
# coding=utf-8
import time
import board
import busio
import smbus2
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


import socket
import sys
from time import sleep
import random
from struct import pack

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

host, port = '141.53.32.82', 65000
server_address = (host, port)
# Create the I2C bus
i2c = busio.I2C(3,2)
#i2c = busio.I2C(0,1,0)
# Create the ADC object using the I2C bus
ads_coffeebox = ADS.ADS1115(i2c)
ads_frenchpress = ADS.ADS1115(i2c, 0x4b)
ads_kettle = ADS.ADS1115(i2c, 0x4a)



# Create single-ended input on channel0
chan0_coffeebox = AnalogIn(ads_coffeebox, ADS.P0)
chan1_coffeebox = AnalogIn(ads_coffeebox, ADS.P1)
chan2_coffeebox = AnalogIn(ads_coffeebox, ADS.P2)
chan3_coffeebox = AnalogIn(ads_coffeebox, ADS.P3)

# French Press Scale
chan0_frenchpress = AnalogIn(ads_frenchpress, ADS.P0)
chan1_frenchpress = AnalogIn(ads_frenchpress, ADS.P1)
chan2_frenchpress = AnalogIn(ads_frenchpress, ADS.P2)
chan3_frenchpress = AnalogIn(ads_frenchpress, ADS.P3)

# kettle Scale 
chan0_kettle = AnalogIn(ads_kettle, ADS.P0)
chan1_kettle = AnalogIn(ads_kettle, ADS.P1)
chan2_kettle = AnalogIn(ads_kettle, ADS.P2)
chan3_kettle = AnalogIn(ads_kettle, ADS.P3)


while True:

    #print("channel 0: ","{:>5}\t{:>5.3f}".format(chan0_coffeebox.value, chan0_coffeebox.voltage))

    # Send Weight to Server
    sum_coffeebox = chan0_coffeebox.value+chan1_coffeebox.value+chan2_coffeebox.value+chan3_coffeebox.value
    sum_frenchpress = chan0_frenchpress.value+chan1_frenchpress.value+chan2_frenchpress.value+chan3_frenchpress.value
    sum_kettle = chan0_kettle.value+chan1_kettle.value+chan2_kettle.value+chan3_kettle.value

    message = pack('3f', sum_coffeebox, sum_frenchpress, sum_kettle)
    sock.sendto(message, server_address)
    time.sleep(1)
