Jetson Train
==========

This repository contains step by step guide to build and train your own model for Jetson Nano or Xavier or any other model. 
You need Ubuntu 18 or higher to follow this guide.

Note that some of the hardware like RFID, GPS, GSM can directly be interfaced with Windows or Linux based machine but some of the modules will require Raspberry Pi or other similar embedded devices.

PyPi: https://pypi.org/project/pyembedded/

Installation:
=============
Package can be installed via pip::

    $ pip3 install pyembedded

Verify if it is installed::

    $ import pyembedded
    $ pyembedded.__version__


RFID Usage:
===========
Run below basic code to get the rfid id

Note: Use port as 'COM1', 'COM2' etc in case of windows machine. Use port as '/dev/ttyUSB0' in case of linux based devices::

    $ from pyembedded.rfid_module.rfid import RFID
    $ rfid = RFID(port='COM3', baud_rate=9600)
    $ print(rfid.get_id())


GPS Usage:
==========
Run below code to get GPS related data

Note: Use port as 'COM1', 'COM2' etc in case of windows machine. Use port as '/dev/ttyUSB0' in case of linux based devices::

    $ from pyembedded.gps_module.gps import GPS
    $ gps = GPS(port='COM3', baud_rate=9600)
    $ print(gps.get_lat_long())

Other methods available::

    $ get_lat_long()
    $ get_time()
    $ get_quality_indicator()
    $ get_no_of_satellites()
    $ get_raw_data()

GSM Usage:
==========
Run below code to interface with GSM SIMCOM module

Note: Use port as 'COM1', 'COM2' etc in case of windows machine. Use port as '/dev/ttyUSB0' in case of linux based devices::

    $ from pyembedded.gsm_module.gsm import GSM
    $ import time
    $ phone = GSM(port='COM3', baud_rate=9600)
    $ if phone.modem_active():
    $     phone.make_call(number='9876543210')
    $     time.sleep(4)
    $     phone.end_ongoing_call()

Other methods available::

    $ phone.make_miss_call(number='9876543210', timeout=5)
    $ phone.get_international_subscriber_identity()
    $ phone.get_modem_serial_number()
    $ phone.get_modem_revision_number()
    $ phone.get_modem_model_no()
    $ phone.get_modem_manufacturer()
    $ phone.get_signal_strength()
    $ phone.send_sms(number="+14691234567", message="Hello World")
    $ phone.read_all_sms()
    $ phone.read_sms_by_msg_id(msg_id=3)

Raspberry Pi Usage:
===================
Run below code to get some useful data from Raspberry Pi

NOTE: Some of the below methods can also run on other linux based os::

    $ from pyembedded.raspberry_pi_tools.raspberrypi import PI
    $ pi = PI()
    $ pi.get_ram_info()
    $ pi.get_disk_space()
    $ pi.get_cpu_usage()
    $ pi.get_connected_ip_addr(network='wlan0')
    $ pi.get_cpu_temp()
    $ pi.get_wifi_status()

