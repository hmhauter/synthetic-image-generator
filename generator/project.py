import cv2
import numpy as np 
from matplotlib import pyplot as plt
import random
import os
import imutils
import time
import re
import logging
import albumentations as A

from dataSplit import DataSplit

class ImageGenerator():
    """
     Image Generator class
    """
    def __init__(self, dataformat):
        self.base_path = ""
        self.dataSplit = DataSplit(self.base_path)
        self.dataFormat = dataformat
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.Rotate(p=0.3, border_mode=cv2.BORDER_REPLICATE),
            A.GaussNoise(var_limit=(5, 20), mean=0, p=0.2),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.Blur(blur_limit=3, p=0.2),
            A.ColorJitter(p=0.2),
            A.Cutout(p=0.5,max_h_size=60, max_w_size=60)
        ],
        bbox_params=A.BboxParams(format=str(dataformat), min_visibility=0.8, label_fields=['class_labels'])
        )
        logging.getLogger().setLevel(logging.INFO)

    
    def blend_gaussian(self, composition, object_mask, blur=7, thickness=5):
        """
         Apply Gaussian Blending to the objet to make it fit into background
         composition: image background plus objects 
         object_mask: segmentation mask for composed image
         blur: kernel size for blurring (the bigger the more blurring), MUST BE AN ODD NUMBER  
         thickness: size of the border where blurring is applied
        """
        # get contours 
        cnts = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # blur the whole object
        composition_gaussian_blur = cv2.GaussianBlur(composition, (blur, blur), 0)
        # create empty mask for contours
        mask = np.zeros(composition.shape, np.uint8)
        mask_contours = cv2.drawContours(mask, cnts, -1, (255,255,255), thickness)

        blended_object = np.where(mask_contours == np.array([255,255,255]), composition_gaussian_blur, composition)
        return blended_object

    def blend_gaussian_double(self, composition, object_mask, blur=7, thickness=5):
        """
         Apply Gaussian Blending to the objet to make it fit into background
         composition: image background plus objects 
         object_mask: segmentation mask for composed image
         blur: kernel size for blurring (the bigger the more blurring), MUST BE AN ODD NUMBER  
         thickness: size of the border where blurring is applied
        """
        # get contours 
        cnts = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # blur the whole object
        composition_gaussian_blur_1 = cv2.GaussianBlur(composition, (3, 3), 0)
        composition_gaussian_blur_2 = cv2.GaussianBlur(composition, (7, 7), 0)
        # create empty mask for contours
        mask_1 = np.zeros(composition.shape, np.uint8)
        mask_2 = np.zeros(composition.shape, np.uint8)
        cv2.drawContours(mask_1, cnts, -1, (255,255,255), 10)
        cv2.drawContours(mask_2, cnts, -1, (255,255,255), 5)

        blended_object = np.where(mask_1 == np.array([255,255,255]), composition_gaussian_blur_1, composition)
        blended_object = np.where(mask_2 == np.array([255,255,255]), composition_gaussian_blur_2, blended_object)

        return blended_object
    
    def get_distractor(self):
        """
         Randomly select a distractor to make sure ANN does not learn edges
        """
        image_files = [f for f in os.listdir(self.base_path+"data\\distractors") if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print("No image files found in the folder.")
        else:
            # Choose a random image file from the list
            random_img = random.choice(image_files)
            return self.base_path+"data\\distractors\\"+random_img

    def generate_shape_distractor(self, background_used, object_mask):
        """
         Generate a shape distractor with the shape of one of the objects and a ransom background
         different to the one that is used 
        """
        backgrounds_folder = self.base_path + "data\\backgrounds" 
        background_files = [f for f in os.listdir(backgrounds_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        isEqual = True
        while isEqual:
            random_background = random.choice(background_files)
            random_background_path = os.path.join(backgrounds_folder, random_background)
            if random_background_path != background_used and "mask" not in random_background_path:
                print(random_background_path)
                isEqual = False

        background_new = cv2.imread(random_background_path)

        max_position_x = background_new.shape[1] - object_mask.shape[1]
        max_position_y = background_new.shape[0] - object_mask.shape[0]
        position_x = random.randint(0, max_position_x)
        position_y = random.randint(0, max_position_y)

        background_cut_out = background_new[position_y:position_y + object_mask.shape[0], position_x:position_x + object_mask.shape[1]]
        background_cut_out[object_mask == 0] = 0

        object_mask_reshaped = np.expand_dims(object_mask, axis=-1)
        new_object = np.concatenate((background_cut_out, object_mask_reshaped), axis=2)
        # new_object = cv2.merge((background_cut_out, object_mask))
        new_object = new_object.astype(np.uint8) 
        return new_object

    def place_object(self, background, background_roi, objects, MODES):
        """
         Main function that decides where to place objects on background and applies blending
         background: RGB image of background 
         background_roi: Black and White segmentation mask for ROI of background image
         objects: array of image paths to objects
         MODES: str, decides the blurring mode
        """
        # GET IMAGES FROM PATH 
        _background_path = background
        background = cv2.imread(background)
        objects = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in objects]
        background_roi = cv2.imread(background_roi, cv2.IMREAD_GRAYSCALE)

        # storage for coordinates of bounding boxes
        coordinates = []
        background_mask = np.where(background_roi == 0)
        combined_mask = np.zeros((background.shape[0], background.shape[1]), dtype=np.uint8)
        segmentation_mask = np.zeros((background.shape[0], background.shape[1]), dtype=np.uint8)
        combined_image = background.copy()
        isDistractor = np.zeros(len(objects))

        # decide if a distractor should be added 
        distractor_decission = random.random()
        # logging.info(f'Should a distractor be added?: {distractor_decission} ')

        # UNCOMMENT TO USE SHAPE DISTRACTORS
        # random_object = random.choice(objects)
        # max_scaling_factor = min(background.shape[0] / random_object.shape[0], background.shape[1] / random_object.shape[1])
        # resize_scale = random.uniform(0.1, 0.25) * max_scaling_factor  
        # object_augmented = cv2.resize(random_object, (int(resize_scale * random_object.shape[1]), int(resize_scale * random_object.shape[0])))
        # # rotate object 
        # rotation_angle = random.randrange(0, 360, 1)
        # object_augmented = imutils.rotate_bound(object_augmented, rotation_angle)
        # # flip object 
        # flip_prob = random.choice(np.arange(-1,3))
        # if(flip_prob != 2):
        #     object_augmented = cv2.flip(object_augmented, flip_prob)
        # # WITH SAM 
        # _tmp_object_mask = np.where(object_augmented[:,:, 3] != 0, 255, 0)
        # _tmp_object_mask = _tmp_object_mask.astype(np.uint8)
        # _tmp_object_mask = np.where(_tmp_object_mask[:,:] != 0, 255, 0)
        # shape_distractor = self.generate_shape_distractor(_background_path, _tmp_object_mask)
        # objects.append(shape_distractor)
        # modes = ["RAW", "GAUSSIAN", "POISSON_NORMAL", "POISSON_MIXED"]
        # MODES.append(random.choice(modes))
        # isDistractor = np.append(isDistractor, 1)

        if distractor_decission < 0.5: # 50 percent chance that a distractor is added 
            distractor_path = self.get_distractor()
            distractor_img = cv2.imread(distractor_path, cv2.IMREAD_UNCHANGED)
            objects.append(distractor_img)
            modes = ["RAW", "GAUSSIAN", "POISSON_NORMAL", "POISSON_MIXED"]
            MODES.append(random.choice(modes))
            isDistractor = np.append(isDistractor, 1)

        
        for iter in range(len(objects)):
            object = objects[iter]
            

            object = cv2.cvtColor(object, cv2.COLOR_BGR2RGBA)

            # RANDOMLY CHANGE COLOR OF OBJECT 
            b, g, r, a = cv2.split(object)
            # Randomly modify each color channel
            random_factor = np.random.uniform(0.5, 1.5)  # Adjust the range based on your preference
            b = np.clip(b * random_factor, 0, 255).astype(np.uint8)
            g = np.clip(g * random_factor, 0, 255).astype(np.uint8)
            r = np.clip(r * random_factor, 0, 255).astype(np.uint8)
   
            # Merge the modified channels back into the final image
            object = cv2.merge([b, g, r, a])

           
            MODE = MODES[iter]

            isValid = False
            while isValid == False: # TODO: here we can do some optimization -> do we have to augment the object completely new every time it does not fit into the background? 
                # debug = background.copy()
                # now the object 
                # first determine size and height of object itself with help of bounding box

                # resize object 
                max_scaling_factor = min(background.shape[0] / object.shape[0], background.shape[1] / object.shape[1])
                resize_scale = random.uniform(0.25, 0.35) * max_scaling_factor # random.uniform(0.1, 0.50) * max_scaling_factor  
                object_augmented = cv2.resize(object, (int(resize_scale * object.shape[1]), int(resize_scale * object.shape[0])))

                # rotate object 
                rotation_angle = random.randrange(0, 360, 1)
                object_augmented = imutils.rotate_bound(object_augmented, rotation_angle)


                # flip object 
                flip_prob = random.choice(np.arange(-1,3))
                if(flip_prob != 2):
                    object_augmented = cv2.flip(object_augmented, flip_prob)

                
                # WITH SAM 
                object_mask = np.where(object_augmented[:,:, 3] != 0, 255, 0)
                object_mask = object_mask.astype(np.uint8)

                # erode the mask such that the cutout gets a bit smaller what means that when we done blur the 
                # painted border we actually blur the cornerline
                kernel = np.ones((3, 3), np.uint8)
                object_mask = cv2.erode(object_mask, kernel, cv2.BORDER_REFLECT) 

                # get the contour 
                object_cnts =  cv2.findContours(object_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                object_cnts = imutils.grab_contours(object_cnts)
                object_augmented = object_augmented[:,:,0:3] 
                
                object_x, object_y, object_w, object_h = cv2.boundingRect(object_cnts[0])
                debug_help = object_mask.copy()
                debug_help = cv2.rectangle(debug_help, (object_x, object_y), (object_x + object_w, object_y + object_h), 255, 5)

                object_extracted = object_augmented[object_y:object_y + object_h, object_x:object_x + object_w]
                object_mask_extracted = object_mask[object_y:object_y + object_h, object_x:object_x + object_w]             

                if object_h % 2 == 0:
                    h_start = int(object_h/2)
                    h_end = int(object_h/2)
                else:
                    h_start = int(object_h/2)
                    h_end = int(object_h/2)+1
                
                if object_w % 2 == 0:
                    w_start = int(object_w/2)
                    w_end = int(object_w/2)
                else:
                    w_start = int(object_w/2)
                    w_end = int(object_w/2)+1

                # first handle background placement
                # MAKE SURE OBJECT CAN FIT 
                index_y = random.randint(0, len(background_mask[0]) - 1)
                index_x = random.randint(0, len(background_mask[1]) - 1)

                background_y = background_mask[0][index_y]
                background_x = background_mask[1][index_x]

                if background_y-h_start >= 0 and background_x-w_start >= 0 and background_y+h_end <= background.shape[0] and background_x+w_end <= background.shape[1]:
                    isValid = True
                else:
                    continue
             
                roi = combined_image[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end]
                roi_mask = background_roi[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end]
                combined_mask_check = combined_mask[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end]

      
                # check if object to be inserted overlaps with any other object 
                # allow an overlap of 25%
                count_positiv = np.count_nonzero(combined_mask_check == 255)
                relation_positiv = count_positiv / combined_mask_check.size

                if roi.shape[0] != 0 and (relation_positiv < 0.25) and np.all(roi_mask == 0):
                    isValid = True
                else:
                    isValid = False
                    continue
   
                # to invert first take mask and convert mask to grayscale
                mask_inv = cv2.bitwise_not(object_mask_extracted)
                # mask out object in ROI
                roi = roi.astype(np.uint8)

                mask_inv = mask_inv.astype(np.uint8)
          
                background_masked = cv2.bitwise_and(roi, roi, mask=mask_inv)

                
                obj_masked = cv2.bitwise_and(object_extracted, object_extracted, mask=object_mask_extracted)
                # combine
                obj_result = cv2.add(obj_masked, background_masked)


                # we also need to update the mask for the combined image 
                combined_mask[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end] = object_mask_extracted
                if isDistractor[iter] == 0:  
                    segmentation_mask[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end] = object_mask_extracted
                # insert object into background image
                combined_image[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end] = obj_result

                if MODE == "POISSON_NORMAL":
                    # normal: gradient of object is preserved
                    poisson_blended = cv2.seamlessClone(object_augmented, background, object_mask, (int(background_x), int(background_y)), cv2.NORMAL_CLONE)
                elif MODE == "POISSON_MIXED":
                    # mixed: gradient is combination of background and object 
                    poisson_blended = cv2.seamlessClone(object_augmented, background, object_mask, (int(background_x), int(background_y)), cv2.MIXED_CLONE)
                if MODE == "POISSON_NORMAL" or MODE == "POISSON_MIXED":
                    # now we need a local mask for the object in the background
                    local_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
                    local_mask[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end] = object_mask_extracted

                    # MASK FOR COMPOSITION
                    combined_cnts = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    combined_cnts = imutils.grab_contours(combined_cnts)

                    combined_mask_contours = np.zeros(poisson_blended.shape, np.uint8)
                    combined_mask_contours = cv2.drawContours(combined_mask_contours, combined_cnts, -1, (255, 255, 255), 3)  

                    combined_image = np.where(combined_mask_contours == np.array([255, 255, 255]), poisson_blended, combined_image)

                elif MODE == "GAUSSIAN":
                    # now we need a local mask for the object in the background
                    local_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
                    local_mask[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end] = object_mask_extracted
                    combined_image = self.blend_gaussian(combined_image, local_mask, blur=5, thickness=6)

                elif MODE == "GAUSSIAN_DOUBLE":
                    local_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
                    local_mask[background_y-h_start:background_y+h_end, background_x-w_start:background_x+w_end] = object_mask_extracted
                    combined_image = self.blend_gaussian_double(combined_image, local_mask, blur=5, thickness=6)

                elif MODE == "RAW":
                    # don't do anything...
                    pass

                if isDistractor[iter] == 0:   
                    # FORMAT (xmin, ymin, xmax, ymax)
                    coordinates.append([background_x - w_start, background_y - h_start, background_x + w_end, background_y + h_end])
            
                    # for debugging bounding box can be visualized here
                    # combined_image = cv2.rectangle(
                    #     combined_image,  
                    #     (background_x - w_start, background_y - h_start), 
                    #     (background_x + w_end, background_y + h_end),      
                    #     (0, 255, 0),  
                    #     thickness=5
                    # )

        return combined_image, coordinates, segmentation_mask


    def generate_images_cutout_obj(self, num_images, SETTING, num_apply_augmentation, path_objects):
        """
         Main function that decides where to place objects on background and applies blending
         num_images: int, decides how many images will be generated (without augmented images)
         SETTING: int, decides which blurring should be applied
         num_apply_augmentation: int, decides how many augmented images should be created from one generated image
        """
        f = open(os.path.join(self.base_path, "implementation\\result\\labels.txt"), 'w')
        objects_folder = path_objects
        backgrounds_folder = self.base_path + "data\\backgrounds" 
        labels_file = os.path.join(objects_folder, "labels.txt")
        # Read the labels from labels.txt
        with open(labels_file, 'r') as labels_file:
            labels_content = labels_file.read()
        # Read the object images
        image_files = [f for f in os.listdir(objects_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Read the backgrounds
        background_files = [f for f in os.listdir(backgrounds_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        file_dict = {}

        # Iterate through the filenames
        for filename in background_files:
            parts = filename.split('-')  # Split the filename by hyphens
            if len(parts) == 3:
                number, prefix, extension = parts
                if prefix == 'calibration' and extension.endswith(('.png', '.jpg', '.jpeg')):
                    # Check if there's a corresponding mask filename
                    mask_filename = f'{number}-mask-Camera.png'
                    if mask_filename in background_files:
                        file_dict[os.path.join(backgrounds_folder, filename)] = os.path.join(backgrounds_folder, mask_filename)
        background_pairs = list(file_dict.items())

        img_counter = 0     # needed to also save tthe augmented images
        for i in range(num_images):
            # Randomly select between 1 to 3 images (with the possibility of selecting the same image multiple times)
            num_images_to_select = random.randint(1, 2)
            selected_images = random.choices(image_files, k=num_images_to_select)
            # randomly select a background
            selected_background_pair = random.choice(background_pairs)

            # Initialize a list to store the selected image paths, their corresponding labels, and their indices
            selected_image_paths = []
            selected_labels = []
            selected_indices = []

            # Extract labels as a list using regular expressions
            labels_list = re.findall(r'\d+', labels_content)

            # Convert labels to integers
            labels_list = [int(label) for label in labels_list]

            # Iterate through the selected images and store their indices
            for index, selected_image in enumerate(image_files):
                if selected_image in selected_images:
                    image_path = os.path.join(objects_folder, selected_image)
                    selected_image_paths.append(image_path)
                    
                    # Get the corresponding label based on the index
                    label = labels_list[index] if index < len(labels_list) else None
                    selected_labels.append(label)
                    selected_indices.append(index)
            # GENERATE THE IMAGE
            # ["RAW", "GAUSSIAN", "POISSON_NORMAL", "POISSON_MIXED"]
            if SETTING == 0:
                MODES = ["RAW"] * len(selected_image_paths)
            elif SETTING == 1:
                MODES = ["GAUSSIAN"] * len(selected_image_paths)
            elif SETTING == 2:
                MODES = random.choices(["POISSON_NORMAL", "POISSON_MIXED"], weights=[0.5, 0.5], k=len(selected_image_paths))
            elif SETTING == 3:
                MODES = random.choices(["RAW", "GAUSSIAN", "POISSON_NORMAL", "POISSON_MIXED"], weights=[0.25, 0.25, 0.25, 0.25], k=len(selected_image_paths))
            elif SETTING == 4:
                MODES = random.choices(["RAW", "GAUSSIAN", "GAUSSIAN_DOUBLE"], weights=[0.4, 0.3, 0.3], k=len(selected_image_paths))
            else:
                logging.error("Please select a valid configuration for the generated images.")
            logging.info(f'Generating Image - Generation {i}')
            combined_image, coordinates, combined_mask = self.place_object(selected_background_pair[0], selected_background_pair[1], selected_image_paths, MODES)

            # TODO: error handling
            if self.dataFormat == "yolo":
                formated_coordinates = self.convert_to_yolo(combined_image, coordinates)
            elif self.dataFormat == "coco":
                formated_coordinates = self.convert_to_coco(coordinates)
            elif self.dataFormat == "pascal_voc":
                formated_coordinates = self.convert_to_pascal_voc(coordinates)
            else:
                logging.error("Please select a valid configuration for the data format.")

            ## AUGMENTATION
            if num_apply_augmentation > 0:
                for _ in range(0, num_apply_augmentation):
                 
                    class_labels = ['.']*len(formated_coordinates)  # create a dummy since it is required - NOT USED!

                    augmented = self.augmentation(image=combined_image, mask=combined_mask, bboxes=formated_coordinates, class_labels=class_labels)
            
                    augmented_image = augmented["image"]
                    augmented_coordinates = augmented["bboxes"]
                    augmented_mask = augmented['mask']

                    # For debugging draw bbox
                    # augmented_image = self.draw_yolo_coordinates(augmented_image, augmented_coordinates)

                    cv2.imwrite(self.base_path+"implementation\\result\\"+str(img_counter)+'_combined.jpg',  cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(self.base_path+"implementation\\result\\segmentation\\"+str(img_counter)+'_mask.jpg',  augmented_mask)
                    for indx in range(np.array(augmented_coordinates).shape[0]):
                        f.write(str(img_counter)+','+str(selected_labels[indx])+','+str(augmented_coordinates[indx][0])+','+str(augmented_coordinates[indx][1])+','+str(augmented_coordinates[indx][2])+','+str(augmented_coordinates[indx][3]))
                        f.write('\n')
                    img_counter += 1


            # For debugging draw bbox
            # combined_image = self.draw_yolo_coordinates(combined_image, yolo_coordinates)

            cv2.imwrite(self.base_path+"implementation\\result\\"+str(img_counter)+'_combined.jpg',  cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self.base_path+"implementation\\result\\segmentation\\"+str(img_counter)+'_mask.jpg',  combined_mask)
            for indx in range(np.array(formated_coordinates).shape[0]):
                f.write(str(img_counter)+','+str(selected_labels[indx])+','+str(formated_coordinates[indx][0])+','+str(formated_coordinates[indx][1])+','+str(formated_coordinates[indx][2])+','+str(formated_coordinates[indx][3]))
                f.write('\n')
            img_counter += 1

    def __get_yolo_segmentation(self):
        """
         To be implemented
        """
        pass

    def draw_yolo_coordinates(self, img, coordinates):
        """
         Draw bounding boxes from YOLO corrdinstes for debugging and validation purpose
         img: combined image
         coordinates: 2D array of bounding box coordinates in YOLO format
        """
        (h, w, _) = img.shape
        for indx in range(np.array(coordinates).shape[0]):
            w_bbox = coordinates[indx][2] * w
            h_bbox = coordinates[indx][3] * h
            x_start = (coordinates[indx][0] * w) - (w_bbox / 2)
            x_end = (coordinates[indx][0] * w) + (w_bbox / 2)
            y_start = (coordinates[indx][1] * h) - (h_bbox / 2)
            y_end = (coordinates[indx][1] * h) + (h_bbox / 2)
            img = cv2.rectangle(
                img,  
                (int(x_start), int(y_start)), 
                (int(x_end), int(y_end)),      
                (0, 255, 0),  
                thickness=5
            )
        return img

    def convert_to_yolo(self, img, coordinates):
        """
         Convert (xmin, ymin, xmax, ymax) to YOLO format scaled (xcenter, ycenter, width, height)
         img: combined image
         coordinates: 2D array of bounding box coordinates (xmin, ymin, xmax, ymax) format
        """
        yolo_coordinates = []
        (h, w, _) = img.shape
        for indx in range(np.array(coordinates).shape[0]):
            w_bbox_center = (coordinates[indx][2]+coordinates[indx][0]) / 2
            h_bbox_center = (coordinates[indx][3]+coordinates[indx][1]) / 2
            w_bbox = (coordinates[indx][2]-coordinates[indx][0]) 
            h_bbox = (coordinates[indx][3]-coordinates[indx][1]) 
            yolo_coordinates.append([w_bbox_center/w, h_bbox_center/h, w_bbox/w, h_bbox/h])
        return yolo_coordinates

    def get_yolo_segmentation():
        # for segmentation we store the normalized boundary 
        pass

    def convert_to_coco(self, coordinates):
        """
         Convert (xmin, ymin, xmax, ymax) to COCO format (xmin, ymin, width, height)
         coordinates: 2D array of bounding box coordinates (xmin, ymin, xmax, ymax) format
        """
        coco_coordinates = []
        for indx in range(np.array(coordinates).shape[0]):
            w_bbox = (coordinates[indx][2]-coordinates[indx][0]) 
            h_bbox = (coordinates[indx][3]-coordinates[indx][1]) 
            coco_coordinates.append([coordinates[indx][0], coordinates[indx][1], w_bbox, h_bbox])
        return coco_coordinates

    def convert_to_pascal_voc(self, coordinates):
        """
         Convert (xmin, ymin, xmax, ymax) to PASCAL VOC format (xmin, ymin, xmax, ymax)
         coordinates: 2D array of bounding box coordinates (xmin, ymin, xmax, ymax) format
        """
        return coordinates

        
if __name__ == "__main__":
    imageGenerator = ImageGenerator("yolo")

    SETTING = 1
    start = time.time()
    path_objects = imageGenerator.base_path + "data\\pens" 
    imageGenerator.generate_images_cutout_obj(1000, SETTING, 1, path_objects)

    end = time.time()
    logging.info(f'Generated images in {end-start} seconds')
    if imageGenerator.dataFormat == "pascal_voc":
        imageGenerator.dataSplit.split_data_pascal_voc(0.6,0.2,0.2, ["Pen","Box"], "implementation\\result")
    elif imageGenerator.dataFormat == "yolo":
        imageGenerator.dataSplit.split_data_yolo(1.0,0.0,0.0, ["Pen"], "implementation\\result")
    elif imageGenerator.dataFormat == "coco":
        imageGenerator.dataSplit.split_data_coco(0.6,0.3,0.1, ["Box", "Leaflet", "Pen"], "implementation\\result")
    end = time.time()
    logging.info(f'Done in {end-start} seconds')
    

# SETTINGS 
# 0: All images are raw
# 1: All images are GAUSSIAN 
# 2: All images are POISSON (50:50)
# 3: mixture with same probability
# 4: mixture with different probabilities ([0.2, 0.5, 0.15, 0.15])

