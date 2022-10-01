import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform

        # determine the image set file to use
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
            
        if not os.path.isfile(image_sets_file):
            image_sets_default = self.root / "ImageSets/Main/default.txt"   # CVAT only saves default.txt

            if os.path.isfile(image_sets_default):
                image_sets_file = image_sets_default
            else:
                raise IOError("missing ImageSet file {:s}".format(str(image_sets_file)))

        # read the image set ID's
        self.ids = self._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            classes = []

            # classes should be a line-separated list
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    classes.append(line.rstrip())

            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            #classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
            
        if logging.root.level is logging.DEBUG:
            logging.debug(f"voc_dataset image_id={image_id}" + ' \n    boxes=' + str(boxes) + ' \n    labels=' + str(labels))

        image = self._read_image(image_id)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
            
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _read_image_ids(self, image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                image_id = line.rstrip()
                
                if len(image_id) <= 0:
                    print('warning - found empty line in {:s}, skipping line'.format(str(image_sets_file)))
                    continue
                    
                if self._get_num_annotations(image_id) > 0:
                    if self._find_image(image_id) is not None:
                        ids.append(line.rstrip())
                    else:
                        print('warning - could not find image {:s} - ignoring from dataset'.format(image_id))
                else:
                    print('warning - image {:s} has no box/labels annotations, ignoring from dataset'.format(image_id))
                    
        return ids

    def _get_num_annotations(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        return len(objects)
        
    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.strip() #.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                
                # retrieve <difficult> element
                is_difficult_obj = object.find('difficult')
                is_difficult_str = '0'

                if is_difficult_obj is not None:    
                    is_difficult_str = object.find('difficult').text

                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
            else:
                print("warning - image {:s} has object with unknown class '{:s}'".format(image_id, class_name))

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _find_image(self, image_id):
        img_extensions = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')
        
        for ext in img_extensions:
            image_file = os.path.join(self.root, "JPEGImages/{:s}{:s}".format(image_id, ext))
            
            if os.path.exists(image_file):
                return image_file
            
        return None
        
    def _read_image(self, image_id):
        image_file = self._find_image(image_id)
        
        if image_file is None:
            raise IOError('failed to load ' + image_file)
            
        image = cv2.imread(str(image_file))
        
        if image is None or image.size == 0:
            raise IOError('failed to load ' + str(image_file))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



