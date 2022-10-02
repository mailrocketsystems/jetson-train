Jetson Train
==========

This repository contains step by step guide to build and train your own model for Jetson Nano or Xavier or any other model. 

You need Ubuntu 18 or higher to follow this guide.


<img src="/img/result.PNG" alt="Alt text" title="Optional title">


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
	
	
You can use any tool to annotate your images. Make sure you are using Pascal VOC format. Save all you annotation xml files in Annotations directory. Once done, create labels.txt file inside your model directory. This file will contain all the labels name from your dataset. For ex, 

    $ object_name1
	$ object_name2
	$ object_name3
	
	
Step 2:
=============
In order to start training, use below command:

    $ python3 train_ssd.py --dataset-type=voc --data=data/{modelname}/ --model-dir=models/{modelname} --batch-size=2 --workers=5 --epochs=500


For ex, if your model name is model0110, then command will be:

    $ python3 train_ssd.py --dataset-type=voc --data=data/model0110/ --model-dir=models/model0110 --batch-size=2 --workers=5 --epochs=500
	

This will start the training. You can adjust the number of epochs/workers as per your requirements.


Step 3:
=============
Once your training completes or if you will loss is very low, you can use results.py script to analyze your result. Running the script, will generate a graph of the training and will also output the best checkpoint.

<img src="/path/to/img.jpg" alt="Alt text" title="Optional title">

