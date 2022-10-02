Jetson Train
==========

This repository contains step by step guide to build and train your own model for Jetson Nano or Xavier or any other model. 

You need Ubuntu 18 or higher to follow this guide.


Installation:
=============
Make sure you have installed below python packages before you start setting up your machine for training ::

    $ pip3 install opencv-python
	$ pip3 install imutils
	$ pip3 install matplotlib
	$ pip3 install torchvision
	$ pip3 install torch
	$ pip3 install boto3
	$ pip3 install pandas
	$ pip3 install urllib3


Step 1:
=============
Clone the repository in your machine. Download and save your test video file in videos directory. Use prepare_dataset script to extract images from your testvideo file. You can adjust the save image counter in prepare_dataset script in order to save more images. Once run, this script will create three directories inside data directory. 

    $ JPEGImages: This directory will have all the images extracted from test video file.
	$ ImageSets: This directory will contain few train and test file.
	$ Annotations: Save all your annotation xml files in this directory.