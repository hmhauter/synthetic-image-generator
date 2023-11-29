from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

coco = COCO('')
img_dir = ''
image_ids = coco.getImgIds()


for image_id in image_ids:

    img = coco.imgs[image_id]

    annotation_ids = coco.getAnnIds(imgIds=image_id)

    anns = coco.loadAnns(annotation_ids)
    print(anns)

    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

    for ann in anns:
        box = ann['bbox']
        tag = str(ann['category_id'])
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[0]+box[2]), int(box[1]+box[3])),
            (55, 255, 155),
            5,
        )
        cv2.putText(image, tag, (int(box[0]), int(box[1]-22)), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 255, 155), 3)

    plt.figure()
    plt.imshow(image)
    plt.show()
