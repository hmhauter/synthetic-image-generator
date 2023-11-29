from ultralytics import YOLO
import torch 
import os
from collections import namedtuple
import cv2
import shutil
import glob
import json
from matplotlib import pyplot as plt
from PIL import Image 

class novo_yolo:
    def __init__(self):

        self.data_folder = ""
        self.name_train = ""
        self.name_val = ""

        self.yolo_folder = ""


    def train_model(self):
        print(torch.cuda.is_available())
        print(torch.version.cuda)

        # Load a model 
        # model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("\\yolov8n.pt")  # load a pretrained model (recommended for training)
        save_path = "" 
        data_path = self.data_folder + "\\data.yaml"

        # Use the model
        model.train(data=data_path, epochs=100, save_dir=save_path, name=self.name_train,patience=20)  # train the model
        path = model.export(format="onnx")  # export the model to ONNX format
        metrics = model.val(save_json=True, save_dir=save_path, name=self.name_val, split='test', iou=0.4)  # evaluate model performance on the validation set
       

        print(metrics.confusion_matrix.matrix)

        print("DONE")

    def plot_bounding_box(self, boxGT, boxPRED):

        # Reformat data points
        box1_x_array = [(boxGT[0]), (boxGT[0])+ (boxGT[2]), (boxGT[0]) + (boxGT[2]), (boxGT[0]), (boxGT[0])]
        box1_y_array = [(boxGT[1]), (boxGT[1]), (boxGT[1])+ (boxGT[3]), (boxGT[1]) + (boxGT[3]), (boxGT[1])]
        box2_x_array = [(boxPRED[0]), (boxPRED[0])+ (boxPRED[2]), (boxPRED[0]) + (boxPRED[2]), (boxPRED[0]), (boxPRED[0])]
        box2_y_array = [(boxPRED[1]), (boxPRED[1]), (boxPRED[1])+ (boxPRED[3]),  (boxPRED[1]) + (boxPRED[3]), (boxPRED[1])]

        plt.rcParams["figure.figsize"] = [5.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.xlim(0, max(max(box1_x_array),max(box2_x_array)) + 0.5)
        plt.ylim(0, max(max(box1_y_array),max(box2_y_array)) + 0.5)
        plt.grid()
        plt.plot(box1_x_array, box1_y_array, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="green", linestyle='--')
        plt.plot(box2_x_array, box2_y_array, marker="o", markersize=2, markeredgecolor="red", markerfacecolor="green", linestyle='--')
        plt.gca().invert_yaxis()
        plt.show()

        
    def predict(self):
        predictions = []

        ## LOAD MODEL
        model = YOLO(self.yolo_folder + '\\best.pt')
        # metrics = model.val(save_json=True, save_dir="Y:\\UHMH\\yolo\\special_course_novo\\runs", name=self.name_val, split='test', iou=0.3)  # evaluate model performance on the validation set
        # print(metrics.confusion_matrix.matrix)
        # SAVE DIRECTORY
        test_set = self.data_folder + '\\test\\images\\'
        prediction_save_path = r''
        prediction_save_name = 'predicted'

        for filename in os.listdir(test_set):
            if filename.endswith(".db"):
                print(f"File with .db extension not allowed: {filename}")
                continue
            f = os.path.join(test_set, filename)
            if os.path.isfile(f):
                print("PREDICT")
                print(test_set+filename)
                predictions.append(model(test_set+filename, save_txt=True, save=True, project=prediction_save_path, name=prediction_save_name, conf=0.5))

        print((predictions))

    def calculate_iou(self, box1, box2, plot = False):
        """
        Calculate Intersection over Union (IoU) for two YOLO format boxes.

        Parameters:
        - box1: List containing [center_x, center_y, width, height] for the first box.
        - box2: List containing [center_x, center_y, width, height] for the second box.

        Returns:
        - IoU: Intersection over Union value.
        """

        if plot:
            self.plot_bounding_box(box1, box2)

        # Convert YOLO format to [x_min, y_min, x_max, y_max] format
        box1 = [
            box1[0] - box1[2] / 2,
            box1[1] - box1[3] / 2,
            box1[0] + box1[2] / 2,
            box1[1] + box1[3] / 2
        ]

        box2 = [
            box2[0] - box2[2] / 2,
            box2[1] - box2[3] / 2,
            box2[0] + box2[2] / 2,
            box2[1] + box2[3] / 2
        ]

        # Calculate intersection coordinates
        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])

        # Calculate area of intersection
        intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

        # Calculate area of union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou

    def _collect_predictions(self):
        #Path of source directory
        src_directory = r''
        src_folders = []
        for folder in os.listdir(src_directory):
            src_folders.append(folder)

        #Path of destination directory  
        dst_directory = r''

        """#clean out destination folder TODO: Right now I clean this folder manually
        files = glob.glob(dst_directory)
        for f in files:
            shutil.rmtree(f)"""

        # Extract file from Source directory and move to Destination directory
        for folder in src_folders:

            #folder name formatting
            correct_folder_path = src_directory + '\\' + folder +  '\\labels'

            for file in os.listdir(correct_folder_path):
                src_file = os.path.join(correct_folder_path, file)  ## TODO: HANDLE THE CASE OF THIS BEING AN EMPTY FILE 
                dest_file = os.path.join(dst_directory, file)
                shutil.move(src_file, dest_file)

        # files = glob.glob(src_directory)
        # for f in files:
        #     shutil.rmtree(f)

    def run_analysis(self):
        #### impelment
        iou_detect_objects = []

        test_set_image_folder = self.data_folder + '\\test\\images\\'
        print(len(os.listdir(test_set_image_folder)), ' test images are found.')
        test_set_label_folder = self.data_folder + '\\test\\labels\\'
        print(len(os.listdir(test_set_label_folder))-1, ' test labels are found.') ## -1 for the classes txt file
        predictions_folder =  "Y:\\UHMH\\yolo\\special_course_novo\\predictions_clean\\"
        print(len(os.listdir(predictions_folder)), ' of the images files have predictions.')
        print(round((len(os.listdir(predictions_folder)) / len(os.listdir(test_set_image_folder)))*100) , '% of test images have predictions.')

        ## merge image to label to bounding box here (they should all have the same name in the directories)
        ## search the predictions folder as they may be less predictions than labels if some did not get labelled. At the end we can print how many actually got labelled
        TP_count, TN_count, FP_count, FN_count = 0,0,0,0
        total_number_of_labels = 0
        total_number_of_predictions = 0
        predicts = []

        for predicted_image_file_name in os.listdir(predictions_folder):

            name, ext = predicted_image_file_name.split(".")

            image_path = test_set_image_folder + name + ".jpg"
            label_path = test_set_label_folder + predicted_image_file_name
            pred_label_path = predictions_folder + predicted_image_file_name
            
            #############3 TODO: NEED TO HANDLE THE CASE OF MULTIPLE LABELS AND PREDS. RN ONLY THE FIRST TWO DOUND ARE COMPARED BUT THEY DONT NECESSARILY MATCH #########
            label = open(label_path, "r").read()
            box = open(pred_label_path, "r").read()
            c_l, x_l, y_l, z_l, w_l, c_p, x_p, y_p, z_p, w_p = [], [], [], [], [], [], [], [], [], []

            labels = label.split("\n")
            for i in range(len(labels)-1): #minus 1 because the last object of the split is always empty
                c, x, y, z, w = labels[i].split(" ")
                c_l.append(float(c))
                x_l.append(float(x))
                y_l.append(float(y))
                z_l.append(float(z))
                w_l.append(float(w))

            boxes = box.split("\n")
            for i in range(len(boxes)-1):
                c, x, y, z, w = boxes[i].split(" ")
                c_p.append(float(c))
                x_p.append(float(x))
                y_p.append(float(y))
                z_p.append(float(z))
                w_p.append(float(w))
                
            predicts.append(Prediction(image_path, labels, boxes ))

            if(len(boxes)>len(labels)):
                # if there is more predictions than labels we have false positives 
                FP_count = FP_count + (len(boxes)-len(labels))

                for lab_index in range(len(labels)-1): 
                    ious = []
                    objects = []
                    for prs_index in range(len(boxes)-1): 
                        if (c_p[prs_index] == c_l[lab_index]):
                            object = Detection(image_path, [float(x_l[lab_index]),float(y_l[lab_index]),float(z_l[lab_index]),float(w_l[lab_index])], 
                                                            [float(x_p[prs_index]),float(y_p[prs_index]),float(z_p[prs_index]),float(w_p[prs_index])] )
                            #iou = bb_intersection_over_union(object.gt, object.pred, plot=False)
                            iou = self.calculate_iou(object.gt, object.pred, plot=False)
                            if 0<iou<1: 
                                ious.append(iou)
                                objects.append(object)
                    if(len(ious) >0):
                        iou_detect_objects.append([objects[ious.index(max(ious))]])
            else:
                # if there is equal amount of labels and predictions we either have true positives or true negatives
                # if there is less predictions than labels we have false negatives 
                for prs_index in range(len(boxes)-1):
                    ious = []
                    objects = []
                    for lab_index in range(len(labels)-1): 
                        if (c_p[prs_index] == c_l[lab_index]):
                            object = Detection(image_path, [float(x_l[lab_index]),float(y_l[lab_index]),float(z_l[lab_index]),float(w_l[lab_index])], 
                                                            [float(x_p[prs_index]),float(y_p[prs_index]),float(z_p[prs_index]),float(w_p[prs_index])] )
                            #iou = bb_intersection_over_union(object.gt, object.pred, plot=False)
                            iou = self.calculate_iou(object.gt, object.pred, plot=False)
                            if 0<iou<1: 
                                ious.append(iou)
                                objects.append(object)
                    if(len(ious) >0):
                        iou_detect_objects.append([objects[ious.index(max(ious))]])
                        TP_count = TP_count + 1

        print("Total Number of correct predictions found: ", len(iou_detect_objects))
        print(len(predicts))

        for labelled_image_file_name in os.listdir(test_set_image_folder):
            pr = True
            for prediction in os.listdir(predictions_folder):
                p1, p2 = labelled_image_file_name.split('.')
                p11, p22 = prediction.split('.')
                if p1 == p11:
                    pr = False

            # print(os.path.join(test_set_image_folder, labelled_image_file_name))
            if pr == True:
                image = cv2.imread(os.path.join(test_set_image_folder, labelled_image_file_name))
                # plt.figure()
                # plt.imshow(image)
                # plt.show()



        for predictionss in predicts:
            # predictionss = detection[0]
            image = cv2.imread(predictionss.image_path)
            img = Image.open(predictionss.image_path) 
            hh = img.height
            ww = img.width

            for i in range(len(predictionss.gts)-1):
                c, x, y, z, w = predictionss.gts[i].split(" ")
                boxA = [float(x)-(float(z)/2), float(y)-(float(w)/2), float(x)+(float(z)/2), float(y)+(float(w)/2)]
                cv2.rectangle(image, [int(boxA[0]*ww), int(boxA[1]*hh)], [int(boxA[2]*ww), int(boxA[3]*hh)], (0, 255, 0), 2)
                    
            for i in range(len(predictionss.preds)-1):
                c, x, y, z, w = predictionss.preds[i].split(" ")
                boxB = [float(x)-(float(z)/2), float(y)-(float(w)/2), float(x)+(float(z)/2), float(y)+(float(w)/2)]
                cv2.rectangle(image,  [int(boxB[0]*ww), int(boxB[1]*hh)], [int(boxB[2]*ww), int(boxB[3]*hh)], (0, 0, 255), 2)
            # plt.figure()
            # plt.imshow(image)
            # plt.show()


        total = 0
        index = 0
        ious = []
        for detection in iou_detect_objects:
            # compute the intersection over union and display it
            iou = self.calculate_iou(detection[0].gt, detection[0].pred, plot=False)
            ious.append(iou)
            total = total + iou
            index = index + 1
        print('Average IoU: ', total/index)

            # loop over the example detections
        for detection in iou_detect_objects:
            detection = detection[0]
            image = cv2.imread(detection.image_path)
            img = Image.open(detection.image_path) 
            h = img.height
            w = img.width

            boxA = [float(detection.gt[0])-(float(detection.gt[2])/2), float(detection.gt[1])-(float(detection.gt[3])/2), float(detection.gt[0])+(float(detection.gt[2])/2), float(detection.gt[1])+(float(detection.gt[3])/2)]
            boxB = [float(detection.pred[0])-(float(detection.gt[2])/2), float(detection.pred[1])-(float(detection.gt[3])/2), float(detection.pred[0])+(float(detection.pred[2])/2), float(detection.pred[1])+(float(detection.gt[3])/2)]

            cv2.rectangle(image, [int(boxA[0]*w), int(boxA[1]*h)], [int(boxA[2]*w), int(boxA[3]*h)], (0, 255, 0), 2)
            cv2.rectangle(image,  [int(boxB[0]*w), int(boxB[1]*h)], [int(boxB[2]*w), int(boxB[3]*h)], (0, 0, 255), 2)

            iou = self.calculate_iou(detection.gt, detection.pred, plot=False)

            cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
            # plt.figure()
            # plt.imshow(image)
            # plt.show()


if __name__ == "__main__":
    novo_yolo = novo_yolo()
    print("START MAIN")
    novo_yolo.train_model()
    Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
    Prediction = namedtuple("Prediction", ["image_path", "gts", "preds"])
    novo_yolo.predict()
    novo_yolo._collect_predictions()
    novo_yolo.run_analysis()
    print("STOP MAIN")