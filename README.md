Jetson Train
==========

This repository contains step by step guide to build and train your own model for Jetson Nano or Xavier or any other model. 

You need Ubuntu 18 or higher to follow this guide. You will also find the output model files in the repo for the model I trained for apples and banana.


<img src="/img/result.PNG" alt="Alt text" title="results">


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


Watch video below: 

<div align="left">
      <a href="https://www.youtube.com/watch?v=fZiY7zUk3TU">
         <img src="https://img.youtube.com/vi/fZiY7zUk3TU/0.jpg" style="width:100%;">
      </a>
</div>


Step 1:
=============
Clone the repository in your machine. Download and save your test video file in videos directory. Use `prepare_dataset` script to extract images from your testvideo file. You can adjust the save image counter in prepare_dataset script in order to save more images. Once run, this script will create three directories inside data directory. 

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
Once your training completes or your loss is very low, you can use `results.py` script to analyze your result. Running the script, will generate a graph of the training and will also output the best checkpoint.

<img src="/img/graph.PNG" alt="Alt text" title="graph">


Step 4:
=============
Make sure you have jetson-inference project installed on your Jetson device. Once you are satisfied with the training results, you can copy the checkpoint file and the labels.txt from your machine to Jetson Nano or Xavier. Place them inside :

    $ /home/username/jetson-inference/python/training/detection/ssd/models
	

Lets first convert checkpoint to onnx format by running below command from `ssd` directory:

    $ python3 onnx_export.py --model-dir=models/model0110
	

This will generate onnx file. From here we can use below command to generate engine file:

    $ detectnet --model=models/model0110/ssd-mobilenet.onnx --labels=models/model0110/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /home/rocket/testvideo.mp4
	

where model0110 is the name of your model. Make sure you replace `/home/rocket/testvideo.mp4` with path of your test video file or webcam/RTSP camera. This command can take upto 10-12mins to complete. 
	
If you want to use the model file which I have trained for apples & banana, you can download it from `mymodels` directory.

