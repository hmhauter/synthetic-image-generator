import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import os
from process_object import Predictor

class BoundingBoxApp:
    def __init__(self, root):
        self.sam_predictor = Predictor()
        self.root = root
        self.root.title("Bounding Box Creator")
        self.root.geometry("900x700")

        self.menu_height = 100

        self.canvas = tk.Canvas(root, width=900, height=700)
        self.canvas.pack()
        self.canvas.create_rectangle(0, 0, 900, self.menu_height, fill="light steel blue")
        

        self.instruction = tk.Label(root, text="Please load an image and select its class. Then drag a bounding box around the object with your mouse.")
        self.instruction.place(x=10, y=5)

        # Dropdown timeeeeee
        self.class_var = tk.StringVar()
        self.class_var.set("Select Class")
        self.class_dropdown = ttk.Combobox(root, textvariable=self.class_var)
        self.label_list = np.genfromtxt("C:\\Users\\UHMH\\Documents\\image-generation\\SAM\\src\\class_labels.csv",
            delimiter=",", dtype=str)
        self.label_list = list(self.label_list)
        self.class_dropdown["values"] = self.label_list
        self.class_dropdown.place(x=10, y=30)

        self.add_class_button = tk.Button(root, text="Add Class", command=self.add_class)
        self.add_class_button.place(x=200, y=30)

        self.open_image_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_image_button.place(x=300, y=30)

        self.save_button = tk.Button(root, text="Segment", command=self.save_bbox)
        self.save_button.place(x=450, y=30)

        self.cancel_button = tk.Button(root, text="Clear Bounding Box", command=self.clear_bbox)
        self.cancel_button.place(x=450, y=60)

        self.cancel_button = tk.Button(root, text="Close", command=self.root.quit)
        self.cancel_button.place(x=620, y=30)

        # WIP: Text input for adding a new class
        # self.new_class_var = tk.StringVar()
        # self.new_class_entry = tk.Entry(root, textvariable=self.new_class_var)
        # self.new_class_entry.place(x=10, y=50)
        # self.new_class_entry.grid_remove()  # Initially hidden

        self.bbox_start_x = None
        self.bbox_start_y = None
        self.bbox_end_x = None
        self.bbox_end_y = None
        self.drawing_bbox = False

    def add_class(self):
        # ToDo: Implement !!!
        self.new_class_entry.grid()

    def save_bbox(self):
        # check if user selecetd a class
        selected_class = self.class_var.get()
        if self.img_path == None:
            tk.messagebox.showwarning(title="Missing image", message="Please create a bounding box in a selected image")
        elif selected_class == "Select Class" or selected_class == "" or selected_class == None or self.img_path == None:
            tk.messagebox.showwarning(title="Missing class", message="Please select the correct class from the drop-down list")
        else:
            # LOADING BOX 
            self.loading_message = self.canvas.create_text(10, 60, text="Loading... Please wait", font=("Helvetica", 14))
            self.root.update() 
            
            input_box = list([ int(self.bbox_start_x/self.scale), int((self.bbox_start_y-self.menu_height)/self.scale), int(self.bbox_end_x/self.scale), int((self.bbox_end_y-self.menu_height)/self.scale)])

            # reset image
            self.canvas.delete("userimg")
            # reset bbox
            self.canvas.delete("bbox")
            self.bbox_start_x = None
            self.bbox_start_y = None
            self.bbox_end_x = None
            self.bbox_end_y = None
            input_box = np.array(input_box)
      
            mask = self.sam_predictor.segment_object(self.img_path, input_box)
            mask_image = np.where(mask, 0, 255).astype(np.uint8)

            # save image mask together with classname
            image_dir = os.path.dirname(self.img_path)
            image_name, image_extension = os.path.splitext(os.path.basename(self.img_path))
            mask_image_name = f"{image_name}_mask{image_extension}"
            text_file_name = f"{image_name}.txt"
            masked_image_path = os.path.join(image_dir, mask_image_name)
            cv2.imwrite(masked_image_path, mask_image)
            text_file_path = os.path.join(image_dir, text_file_name)
            with open(text_file_path, 'w') as text_file:
                text_file.write(selected_class)

        self.canvas.delete(self.loading_message)
        print(self.img_path)
        print("Save BB with class: " + selected_class)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])

        if file_path:
            self.img_path = file_path
            self.display_image(file_path)
            # Automatically start bounding box creation
            self.start_bbox()

    def start_bbox(self):
        self.canvas.bind("<ButtonPress-1>", self.on_bbox_start)
        self.canvas.bind("<B1-Motion>", self.on_bbox_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_bbox_end)
        self.drawing_bbox = True

    def on_bbox_start(self, event):
        self.bbox_start_x = event.x
        self.bbox_start_y = event.y

    def on_bbox_drag(self, event):
        if self.bbox_start_x is not None and self.bbox_start_y is not None:
            # Draw a rectangle as the user drags the mouse
            self.canvas.delete("bbox")
            self.canvas.create_rectangle(
                self.bbox_start_x,
                self.bbox_start_y,
                event.x,
                event.y,
                outline="red",
                width=2,
                tags="bbox"
            )

    def clear_bbox(self):
        self.canvas.delete("bbox")
        self.bbox_start_x = None
        self.bbox_start_y = None
        self.bbox_end_x = None
        self.bbox_end_y = None
        self.start_bbox()


    def on_bbox_end(self, event):
        self.bbox_end_x = event.x
        self.bbox_end_y = event.y
        self.drawing_bbox = False
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def display_image(self, image_path):
        image = Image.open(image_path)
        width, height = image.size 
        self.scale = 600 / height
        newsize = (int(width*self.scale), int(height*self.scale))   # has to be width height
        image = image.resize(newsize)
        self.photo = ImageTk.PhotoImage(image=image, size=image.size)
        self.canvas.create_image(0, 100, anchor=tk.NW, image=self.photo, tag="userimg")

if __name__ == "__main__":
    root = tk.Tk()
    app = BoundingBoxApp(root)
    root.mainloop()
