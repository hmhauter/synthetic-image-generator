# synthetic-image-generator
Generate synthetic image datasets from background and foreground images and export them in common Deep Learning formats.

## Installation
Run the command `pip install -r requirements.txt` to install the required Python packages with a Python version > 3.6. Additionally, install Node v.18.xx.

### Run Backend
Run the backend with `python manage.py runserver`
### Run Frontend
Run the frontend with `npm start` then you can access the server via http://localhost:3000/
### Run image generator from python file
Images can also directly be enerated from the python file `project.py`. The following settings can be given to the functions:
- ImageGenerator((str)dataset format yolo/coco/pascal_voc)
- generate_images_cutout_obj((int)number of images to generate, (int)blending setting 0-4, (int)how many augmented images should be created from one generated image, (str)path to objects)
- split_data_yolo((float)split train, (float)split validate, (float)split test, (str[])class names, (str)path where images are stored that should be split)