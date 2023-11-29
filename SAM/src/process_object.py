import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


import sys
sys.path.append(r'C:\Users\UHMH\Documents\image-generator\SAM\src')

from segment_anything import sam_model_registry, SamPredictor

class Predictor():
    def __init__(self):
        sam_checkpoint = "C:\\Users\\UHMH\\Documents\\synthetic-image-creation\\SAM\\src\\sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        # print(f'X0: {x0} and Y0: {y0} and w: {w} and h: {h}')
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
    
    
    def segment_object(self, image_path, input_box):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        # input_box = np.array([425, 600, 700, 875])
        
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )


        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # self.show_mask(masks[0], plt.gca())
        # self.show_box(input_box, plt.gca())
        # plt.axis('off')
        # plt.show()

        return masks[0]
    
if __name__ == "__main__":
    sam = Predictor()
    sam.segment_object("C:\\Users\\UHMH\\Documents\\synthetic-image-creation\\SAM\\src\\images\\truck.jpg", np.array([425, 600, 700, 875]))
