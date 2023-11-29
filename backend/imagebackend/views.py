import base64
from django.shortcuts import render
from PIL import Image
import numpy as np
import json
from io import BytesIO
from matplotlib import pyplot as plt
import cv2

from django.core.files.base import ContentFile
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

import sys
sys.path.append(r'C:\Users\UHMH\Documents\image-generator')
from SAM.src.process_object import Predictor
from .serializers import TextSerializer

from .models import UploadedImage
from .serializers import UploadedImageSerializer

class GeneratorView(APIView):
    def post(self, request, format=None):
        received_data = request.data.get('data', {})
        print(received_data)
        return Response({'message': 'Data received successfully'}, status=status.HTTP_200_OK)

class TextView(APIView):
    def get(self, request, format=None):
        # You can retrieve text from a model or define it here
        text = "This is the text you want to send to the frontend."
        serializer = text
        return Response(serializer)
    
    def post(self, request, format=None):
        # Assuming the string is sent in the 'data' field of the JSON payload
        received_data = request.data.get('data', '')
        # Process the received string (e.g., save to database, perform some action, etc.)
        print(received_data)
        return Response({'message': 'String received successfully'}, status=status.HTTP_200_OK)

class SegmentView(APIView):

    sam_predictor = Predictor()
    def resize_bbox(self, bbox_old, size_resized, size_original):
        print("START")
        print(size_original[0])
        print(size_resized[0])
        scale_x = size_original[0] / float(size_resized[0])
        scale_y = size_original[1] / float(size_resized[1])

        print("GET SCALE")

        # Calculate the actual coordinates in the original image
        x_original = float(bbox_old[0]) * scale_x
        y_original = float(bbox_old[1]) * scale_y
        width_original = float(bbox_old[2]) * scale_x
        height_original = float(bbox_old[3]) * scale_y

        return np.array([int(x_original), int(y_original), int(x_original+width_original), int(y_original+height_original)])



    def post(self, request, format=None):
        try:
            received_data = request.data.get('data', {})
            print(received_data)
            image_data = received_data.get('image', '')
            size_data = received_data.get('size', [0, 0])
            bbox_data = received_data.get('bbox', [0, 0, 0, 0])

            size_data = json.loads(size_data)
            bbox_data = json.loads(bbox_data)

            format, imgstr = image_data.split(';base64,') 

            image_data = base64.b64decode(imgstr)
            image = Image.open(BytesIO(image_data))

            print("SIZE: ", image.size)

            # Display the image
            image.show()
            image.save('output_image.jpg')
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            input_box = self.resize_bbox(bbox_data, size_data, image.size)
            print(input_box)
            mask = self.sam_predictor.segment_object(r'C:\Users\UHMH\Documents\image-generator\backend\output_image.jpg', input_box)
            print(mask)
            mask_image = np.where(mask, 0, 255).astype(np.uint8)
            print(mask_image)
            image_array = np.array(image)

            alpha_channel = np.ones_like(image_array[:, :, 0]) * 255
            print(alpha_channel)
            image_array = np.dstack((image_array, alpha_channel))
            print(image_array)

            # Apply the mask to the alpha channel
            alpha_channel[mask_image == 255] = 0
            # Apply the modified alpha channel to the image
            image_array[:, :, 3] = alpha_channel
            result_image = Image.fromarray(image_array)
            result_image.show()
            result_image.save("C:\\Users\\UHMH\Documents\\image-generator\\data\\pens\\pen_37.png")
            print("Bounding Box Data:", bbox_data)
            counter += 1

            buffered = BytesIO()
            result_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return Response({'message': 'Data received successfully','image':img_str}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class UploadImageView(APIView):
    print("Call upload image")
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)