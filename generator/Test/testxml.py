# implementation inspireb by Medium post:
# https://piyush-kulkarni.medium.com/visualize-the-xml-annotations-in-python-c9696ba9c188

import os
import cv2
import xml.dom.minidom
from matplotlib import pyplot as plt

# set path of images you want to test
dataset_path = ""

image_path = dataset_path
annotation_path = dataset_path

files_name = os.listdir(image_path)
for filename_ in files_name:
    filename, extension = os.path.splitext(filename_)
    img_path = image_path + filename + ".jpg"
    xml_path = annotation_path + filename + ".xml"
    print(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found.")
        pass
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    objects = dom.getElementsByTagName("object")
    i = 0
    for object in objects:
        tag = str(root.getElementsByTagName("name")[i].childNodes[0].data)
        bndbox = root.getElementsByTagName("bndbox")[i]
        xmin = bndbox.getElementsByTagName("xmin")[0]
        ymin = bndbox.getElementsByTagName("ymin")[0]
        xmax = bndbox.getElementsByTagName("xmax")[0]
        ymax = bndbox.getElementsByTagName("ymax")[0]
        xmin_data = float(xmin.childNodes[0].data)
        ymin_data = float(ymin.childNodes[0].data)
        xmax_data = float(xmax.childNodes[0].data)
        ymax_data = float(ymax.childNodes[0].data)
        print(object)
        print(tag)
        print(img.shape)
        print(xmin_data)
        print(ymin_data)

        cv2.rectangle(
            img,
            (int(xmin_data), int(ymin_data)),
            (int(xmax_data), int(ymax_data)),
            (55, 255, 155),
            5,
        )
        cv2.putText(img, tag, (int(xmin_data), int(ymin_data-22)), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 255, 155), 3)

        i += 1
    plt.figure()
    plt.imshow(img)
    plt.show()
