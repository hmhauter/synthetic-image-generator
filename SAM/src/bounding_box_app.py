import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class BoundingBoxApp:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path

        image = Image.open(image_path)
        width, height = image.size 
        scale = 0.3
        newsize = (int(width*scale), int(height*scale))
        self.image = image.resize(newsize)

        self.cv_image = cv2.imread(image_path)
        # cv2.imshow("Image", self.cv_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_click)

        self.save_button = tk.Button(root, text="Save Bounding Box", command=self.save_bounding_box)
        self.save_button.pack(anchor=tk.NW)

        self.start_x, self.start_y = None, None
        self.end_x, self.end_y = None, None
        self.rect = None


    def resize_image(self, image, width=None, height=None):
        if width is not None and height is not None:
            return cv2.resize(image, (width, height))
        elif width is not None:
            aspect_ratio = float(image.shape[1]) / float(image.shape[0])
            new_height = int(width / aspect_ratio)
            return cv2.resize(image, (width, new_height))
        elif height is not None:
            aspect_ratio = float(image.shape[1]) / float(image.shape[0])
            new_width = int(height * aspect_ratio)
            return cv2.resize(image, (new_width, height))
        else:
            return image
        
    def save_bounding_box(self):
        print("Save was clicked")
        if self.start_x is not None and self.end_x is not None:
            x1, y1, x2, y2 = self.start_x, self.start_y, self.end_x, self.end_y
            # Save the coordinates of the bounding box or process it as needed
            print(f"Bounding box coordinates: ({x1}, {y1}) - ({x2}, {y2})")
            self.canvas.destroy()            
            self.root.quit()
            self.root.destroy()
            exit()
        else:
            print("No bounding box selected.")

        

    def on_mouse_click(self, event):
        if self.start_x is None:
            self.start_x, self.start_y = event.x, event.y
        else:
            self.end_x, self.end_y = event.x, event.y
            self.draw_rectangle()

    def draw_rectangle(self):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.end_x, self.end_y,
            outline="red", width=2
        )

    def save_bounding_box(self):
        if self.start_x is not None and self.end_x is not None:
            x1, y1, x2, y2 = self.start_x, self.start_y, self.end_x, self.end_y
            # Save the coordinates of the bounding box or process it as needed
            print(f"Bounding box coordinates: ({x1}, {y1}) - ({x2}, {y2})")
        else:
            print("No bounding box selected.")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('700x700')
    root.title("Bounding Box Creator")

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])

    if file_path:
        app = BoundingBoxApp(root, file_path)
        app.root.mainloop()
        app.root.destroy()
