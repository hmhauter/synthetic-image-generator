import cv2
import random
import os
from datetime import datetime
import shutil
import logging
import yaml
import json
import xml.etree.ElementTree as ET


class DataSplit:
    """
     Data Split class
     Format pascal voc, yolo or coco
     Data is randomly distributed in train, test and validation folders
    """
    def __init__(self, base_path) -> None:
        self.base_path = base_path
        logging.getLogger().setLevel(logging.INFO)

    def split_data_pascal_voc(self, train_ratio, validation_ratio, test_ratio, classes, pth):
        folder_path = self.base_path+pth
        f = open(folder_path+"\\labels.txt", 'r')
        img_list = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_list.append(filename)

        img_list = sorted(img_list)

        ratios = [train_ratio, validation_ratio, test_ratio]
        num_elements = [int(ratio * len(img_list)) for ratio in ratios]
        numbers = list(range(len(img_list)))
        random.shuffle(numbers)

        # Split the shuffled list into three parts based on the calculated counts
        train_selection = numbers[:num_elements[0]]
        val_selection = numbers[num_elements[0]:num_elements[0] + num_elements[1]]
        test_selection = numbers[num_elements[0] + num_elements[1]:]

        label_data = []
        labels = f.readlines()
        for line in labels:
            values = line.strip().split(',')
            values = [float(value) for value in values]
            label_data.append(values)
        f.close()
        # print(label_data)
        # create folders 
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        os.mkdir(self.base_path + dt_string)
        os.mkdir(self.base_path + dt_string + "\\test")
        os.mkdir(self.base_path + dt_string + "\\train")
        os.mkdir(self.base_path + dt_string + "\\valid")

        # first the train data
        for train_indx in train_selection:
            train_img_indx = img_list.index(str(train_indx)+"_combined.jpg")
            train_img = img_list[train_img_indx]
            source_file = folder_path + '\\' + train_img
            train_img_read = cv2.imread(source_file)
            destination_file = self.base_path + dt_string + "\\train\\" + train_img
            shutil.move(source_file, destination_file)
            xml_file_path = self.base_path + dt_string + "\\train\\" + train_img.replace(".jpg", ".xml")
            xml_tree = self.__create_pascal_voc_xml(train_img_read, classes, label_data, train_indx, str(train_indx)+"_combined.jpg")
            xml_tree.write(xml_file_path)

        
        # then the test data
        for val_indx in val_selection:
            val_img_indx = img_list.index(str(val_indx)+"_combined.jpg")
            val_img = img_list[val_img_indx]
            source_file = folder_path + '\\' + val_img
            val_img_read = cv2.imread(source_file)
            destination_file = self.base_path + dt_string + "\\valid\\" + val_img
            shutil.move(source_file, destination_file)
            xml_file_path = self.base_path + dt_string + "\\valid\\" + val_img.replace(".jpg", ".xml")
            xml_tree = self.__create_pascal_voc_xml(val_img_read, classes, label_data, val_indx, str(val_indx)+"_combined.jpg")
            xml_tree.write(xml_file_path)

        # then the test data
        for test_indx in test_selection:
            test_img_indx = img_list.index(str(test_indx)+"_combined.jpg")
            test_img = img_list[test_img_indx]
            source_file = folder_path + '\\' + test_img
            test_img_read = cv2.imread(source_file)
            destination_file = self.base_path + dt_string + "\\test\\" + test_img
            shutil.move(source_file, destination_file)
            xml_file_path = self.base_path + dt_string + "\\test\\" + test_img.replace(".jpg", ".xml")
            xml_tree = self.__create_pascal_voc_xml(test_img_read, classes, label_data, test_indx, str(test_indx)+"_combined.jpg")
            xml_tree.write(xml_file_path)



    def __create_pascal_voc_xml(self, image, class_list, label_data, indx, img_str):
        (h, w, d) = image.shape
        # create the root element
        annotation = ET.Element("annotation")

        # create the child elements
        folder = ET.SubElement(annotation, "folder")
        folder.text = ""

        filename = ET.SubElement(annotation, "filename")
        filename.text = img_str

        path = ET.SubElement(annotation, "path")
        path.text = img_str

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "nn_image_generator"

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(w)
        height = ET.SubElement(size, "height")
        height.text = str(h)
        depth = ET.SubElement(size, "depth")
        depth.text = str(d)

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        for row in [row for row in label_data if row[0] == indx]:
            object = ET.SubElement(annotation, "object")
            name = ET.SubElement(object, "name")
            name.text = class_list[int(row[1])]
            pose = ET.SubElement(object, "pose")
            pose.text = "Unspecified"
            bndbox = ET.SubElement(object, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(row[2])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(row[4])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(row[3])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(row[5])

        # create the XML tree and write it to a file
        tree = ET.ElementTree(annotation)
        return tree

    def split_data_coco(self, train_ratio, validation_ratio, test_ratio, classes, pth):     
        folder_path = self.base_path+pth
        f = open(folder_path+"\\labels.txt", 'r')
        img_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_list.append(filename)
        img_list = sorted(img_list)

        label_data = []
        labels = f.readlines()
        for line in labels:
            values = line.strip().split(',')
            values = [float(value) for value in values]
            label_data.append(values)
        f.close()

        ratios = [train_ratio, validation_ratio, test_ratio]
        num_elements = [int(ratio * len(img_list)) for ratio in ratios]
        numbers = list(range(len(img_list)))
        random.shuffle(numbers)
        # Split the shuffled list into three parts based on the calculated counts
        train_selection = numbers[:num_elements[0]]
        val_selection = numbers[num_elements[0]:num_elements[0] + num_elements[1]]
        test_selection = numbers[num_elements[0] + num_elements[1]:]

        # create folders needed for COCO
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M")
        formatted_date_coco = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        year = now.year

        os.mkdir(self.base_path + dt_string)
        os.mkdir(self.base_path + dt_string + "\\test")
        os.mkdir(self.base_path + dt_string + "\\train")
        os.mkdir(self.base_path + dt_string + "\\valid")

        # create JSON
        coco_data_info = {
            "year": year, 
            "version": "1", 
            "description": "Synthetic Image generator - COCO Format", 
            "contributor": "UHMH", 
            "date_created": formatted_date_coco,
        }

        coco_data_category = []
        for c in range(len(classes)):
            coco_data_category.append({
                "id": c,
                "name": classes[c],
                "supercategory": "object"
            })

        # TRAIN
        coco_data_images_train, coco_data_annotations_train = self.__create_json_coco(train_selection, img_list, label_data, folder_path, formatted_date_coco, dt_string, "train")
        json_file_path = self.base_path + dt_string + "\\train\\" + "_annotations.coco.json"
        train_json = {
            "info": coco_data_info,
            "categories": coco_data_category,
            "images": coco_data_images_train,
            "annotations": coco_data_annotations_train
        }
        with open(json_file_path, "w") as write_file:
            write_file.write(json.dumps(train_json))

        # TEST 
        coco_data_images_test, coco_data_annotations_test = self.__create_json_coco(test_selection, img_list, label_data, folder_path, formatted_date_coco, dt_string, "test")
        json_file_path = self.base_path + dt_string + "\\test\\" + "_annotations.coco.json"
        test_json = {
            "info": coco_data_info,
            "categories": coco_data_category,
            "images": coco_data_images_test,
            "annotations": coco_data_annotations_test
        }
        with open(json_file_path, "w") as write_file:
            write_file.write(json.dumps(test_json))

        # VALIDATE
        coco_data_images_val, coco_data_annotations_val = self.__create_json_coco(val_selection, img_list, label_data, folder_path, formatted_date_coco, dt_string, "valid")
        json_file_path = self.base_path + dt_string + "\\valid\\" + "_annotations.coco.json"
        valid_json = {
            "info": coco_data_info,
            "categories": coco_data_category,
            "images": coco_data_images_val,
            "annotations": coco_data_annotations_val
        }
        with open(json_file_path, "w") as write_file:
            write_file.write(json.dumps(valid_json))

    def __create_segmentation_coco(self, selection, img_list, label_data, folder_path, formatted_date_coco, dt_string, split_str, segmentation_mask):
        """
         Coco dataset has segmentation masks RLE encoded 
         img: combined image
         coordinates: 2D array of bounding box coordinates in YOLO format
        """
        pass

    def __create_json_coco(self, selection, img_list, label_data, folder_path, formatted_date_coco, dt_string, split_str):
        coco_data_images = []
        coco_data_annotations = []
        counter = 0
        for _indx in selection:
            train_img_indx = img_list.index(str(_indx)+"_combined.jpg")
            train_img = img_list[train_img_indx]
            source_file = folder_path + '\\' + train_img
            train_img_read = cv2.imread(source_file)
            (h, w, _) = train_img_read.shape
            destination_file = self.base_path + dt_string + "\\" + split_str +"\\" + train_img
            shutil.move(source_file, destination_file)
            # write images
            coco_data_images.append({
                "id": _indx,
                "file_name": str(_indx)+"_combined.jpg",
                "height": h,
                "width": w,
                "date_captured": formatted_date_coco
            })

            # write annotations
            for row in [row for row in label_data if row[0] == _indx]:
                coco_data_annotations.append({
                    "id": counter,
                    "image_id": _indx,
                    "category_id": int(row[1]),   
                    "bbox": row[2:],    
                    "area": float(h*w),
                })
                counter += 1
        return coco_data_images, coco_data_annotations

    def split_data_yolo(self, train_ratio, validation_ratio, test_ratio, classes, pth):
        folder_path = self.base_path+pth
        f = open(folder_path+"\\labels.txt", 'r')
        img_list = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_list.append(filename)

        img_list = sorted(img_list)

        ratios = [train_ratio, validation_ratio, test_ratio]
        num_elements = [int(ratio * len(img_list)) for ratio in ratios]
        numbers = list(range(len(img_list)))
        random.shuffle(numbers)
        # print(len(img_list))
        # print(img_list)
        # Split the shuffled list into three parts based on the calculated counts
        train_selection = numbers[:num_elements[0]]
        val_selection = numbers[num_elements[0]:num_elements[0] + num_elements[1]]
        test_selection = numbers[num_elements[0] + num_elements[1]:]
        # print("TRAIN")
        # print(train_selection)
        # print("VAL")
        # print(val_selection)
        # print("TEST")
        # print(test_selection)
        
        label_data = []
        labels = f.readlines()
        for line in labels:
            values = line.strip().split(',')
            values = [float(value) for value in values]
            label_data.append(values)
        f.close()
        # print(label_data)
        # create folders 
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        os.mkdir(self.base_path + dt_string)
        os.mkdir(self.base_path + dt_string + "\\test")
        os.mkdir(self.base_path + dt_string + "\\test\\images")
        os.mkdir(self.base_path + dt_string + "\\test\\labels")
        os.mkdir(self.base_path + dt_string + "\\train")
        os.mkdir(self.base_path + dt_string + "\\train\\images")
        os.mkdir(self.base_path + dt_string + "\\train\\labels")
        os.mkdir(self.base_path + dt_string + "\\valid")
        os.mkdir(self.base_path + dt_string + "\\valid\\images")
        os.mkdir(self.base_path + dt_string + "\\valid\\labels")

        # create YML file for training
        yml_data = {
            'train': self.base_path + dt_string + "\\train\\images",
            'test': self.base_path + dt_string + "\\test\\images",
            'val': self.base_path + dt_string + "\\valid\\images",
            'nc': len(classes),
            'names': classes
        }
        yml_file_path = self.base_path + dt_string + '\\data.yaml'
        with open(yml_file_path, 'w') as file:
            yaml.dump(yml_data, file, default_flow_style=False)

        # first the train data
        for train_indx in train_selection:
            train_img_indx = img_list.index(str(train_indx)+"_combined.jpg")
            train_img = img_list[train_img_indx]
            source_file = folder_path + '\\' + train_img
            destination_file = self.base_path + dt_string + "\\train\\images\\" + train_img
            shutil.move(source_file, destination_file)
            txt_file_path = self.base_path + dt_string + "\\train\\labels\\" + train_img.replace(".jpg", ".txt")
            with open(txt_file_path, 'w') as train_file:
                for row in [row for row in label_data if row[0] == train_indx]:
                    for element in row[1:]:
                        train_file.write(str(element) + ' ')
                    train_file.write('\n')
        
        # then the test data
        for val_indx in val_selection:
            val_img_indx = img_list.index(str(val_indx)+"_combined.jpg")
            val_img = img_list[val_img_indx]
            source_file = folder_path + '\\' + val_img
            destination_file = self.base_path + dt_string + "\\valid\\images\\" + val_img
            shutil.move(source_file, destination_file)
            txt_file_path = self.base_path + dt_string + "\\valid\\labels\\" + val_img.replace(".jpg", ".txt")
            with open(txt_file_path, 'w') as val_file:
                for row in [row for row in label_data if row[0] == val_indx]:
                    for element in row[1:]:
                        val_file.write(str(element) + ' ')
                    val_file.write('\n')

        # then the test data
        for test_indx in test_selection:
            test_img_indx = img_list.index(str(test_indx)+"_combined.jpg")
            test_img = img_list[test_img_indx]
            source_file = folder_path + '\\' + test_img
            destination_file = self.base_path + dt_string + "\\test\\images\\" + test_img
            shutil.move(source_file, destination_file)
            txt_file_path = self.base_path + dt_string + "\\test\\labels\\" + test_img.replace(".jpg", ".txt")
            with open(txt_file_path, 'w') as test_file:
                for row in [row for row in label_data if row[0] == test_indx]:
                    for element in row[1:]:
                        test_file.write(str(element) + ' ')
                    test_file.write('\n')

    