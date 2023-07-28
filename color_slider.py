import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox

class ColorSlider(QWidget):
    def __init__(self):
        super().__init__()

        self.image_paths = []  # Initialize image_paths as an empty list

        self.initUI()

    def initUI(self):
        # Set up the layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create the button to open the file dialog
        self.open_button = QPushButton('Open Image')
        self.open_button.clicked.connect(self.openImage)
        main_layout.addWidget(self.open_button)

        # Create the button to move to the previous image
        self.previous_button = QPushButton('Previous Image')
        self.previous_button.clicked.connect(self.previousImage)
        main_layout.addWidget(self.previous_button)

        # Create the button to move to the next image
        self.next_button = QPushButton('Next Image')
        self.next_button.clicked.connect(self.nextImage)
        main_layout.addWidget(self.next_button)

        # Create the button to save the filtered image
        self.save_button = QPushButton('Save Image')
        self.save_button.clicked.connect(self.saveImage)
        main_layout.addWidget(self.save_button)

        # Create the sliders for HSV values
        slider_layout = QVBoxLayout()

        self.h_min_slider = QSlider(Qt.Horizontal)
        self.s_min_slider = QSlider(Qt.Horizontal)
        self.v_min_slider = QSlider(Qt.Horizontal)

        self.h_max_slider = QSlider(Qt.Horizontal)
        self.s_max_slider = QSlider(Qt.Horizontal)
        self.v_max_slider = QSlider(Qt.Horizontal)

        # Set the range and initial values for the sliders
        self.h_min_slider.setRange(0, 180)
        self.s_min_slider.setRange(0, 255)
        self.v_min_slider.setRange(0, 255)

        self.h_max_slider.setRange(0, 180)
        self.s_max_slider.setRange(0, 255)
        self.v_max_slider.setRange(0, 255)

        self.h_min_slider.setValue(0)
        self.s_min_slider.setValue(0)
        self.v_min_slider.setValue(0)

        self.h_max_slider.setValue(180)
        self.s_max_slider.setValue(255)
        self.v_max_slider.setValue(255)

        # Create labels for the sliders
        h_min_label = QLabel("H_min")
        s_min_label = QLabel("S_min")
        v_min_label = QLabel("V_min")

        h_max_label = QLabel("H_max")
        s_max_label = QLabel("S_max")
        v_max_label = QLabel("V_max")

        # Connect the sliders to the color update function
        self.h_min_slider.valueChanged.connect(self.updateColor)
        self.s_min_slider.valueChanged.connect(self.updateColor)
        self.v_min_slider.valueChanged.connect(self.updateColor)

        self.h_max_slider.valueChanged.connect(self.updateColor)
        self.s_max_slider.valueChanged.connect(self.updateColor)
        self.v_max_slider.valueChanged.connect(self.updateColor)

        # Add the labels and sliders to the layout
        slider_layout.addWidget(h_min_label)
        slider_layout.addWidget(self.h_min_slider)

        slider_layout.addWidget(s_min_label)
        slider_layout.addWidget(self.s_min_slider)

        slider_layout.addWidget(v_min_label)
        slider_layout.addWidget(self.v_min_slider)

        slider_layout.addWidget(h_max_label)
        slider_layout.addWidget(self.h_max_slider)

        slider_layout.addWidget(s_max_label)
        slider_layout.addWidget(self.s_max_slider)

        slider_layout.addWidget(v_max_label)
        slider_layout.addWidget(self.v_max_slider)

        # Add the slider layout to the main layout
        main_layout.addLayout(slider_layout)

        # Create the image labels
        self.original_image_label = QLabel(self)
        main_layout.addWidget(self.original_image_label)

        self.filtered_image_label = QLabel(self)
        main_layout.addWidget(self.filtered_image_label)

        # Create the image name label
        self.image_name_label = QLabel(self)
        main_layout.addWidget(self.image_name_label)

        # Initialize image variables
        self.original_image = None
        self.filtered_image = None
        self.current_image_index = 0

    def openImage(self):
        # Open the file dialog and get the selected folder path
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(self, 'Open Folder')

        if folder_path:
            # Get a list of image files within the selected folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            if image_files:
                self.image_paths = [os.path.join(folder_path, file) for file in image_files]
                self.current_image_index = 0
                self.loadImage()
            
    def nextImage(self):
        if len(self.image_paths) > 0:
            if len(self.image_paths) > 1:
                self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.loadImage()

    def previousImage(self):
        if len(self.image_paths) > 0:
            if len(self.image_paths) > 1:
                self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.loadImage()

    def loadImage(self):
        # Load the image and set it on the original image label
        image_path = self.image_paths[self.current_image_index]
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, width * channel, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.original_image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        self.original_image = image
        self.updateColor()

        # Display the image name
        image_name = os.path.basename(image_path)
        self.image_name_label.setText(f"Original Image: {image_name}")

    def saveImage(self):
        if hasattr(self, 'filtered_image'):
            # Use the last directory if available, otherwise use the current directory
            initial_dir = self.last_save_dir if hasattr(self, 'last_save_dir') else ''

            # Get the original image name and extension
            image_path = self.image_paths[self.current_image_index]
            image_name = os.path.basename(image_path)
            image_name, image_ext = os.path.splitext(image_name)

            # Get the HSV filter values
            h_min = self.h_min_slider.value()
            s_min = self.s_min_slider.value()
            v_min = self.v_min_slider.value()

            h_max = self.h_max_slider.value()
            s_max = self.s_max_slider.value()
            v_max = self.v_max_slider.value()

            # Construct the new file name
            new_file_name = f"{image_name}_hls_{h_min}_{s_min}_{v_min}_{h_max}_{s_max}_{v_max}{image_ext}"

            # Check if the current output folder is the same as the last one
            if initial_dir and os.path.dirname(self.image_paths[self.current_image_index]) == self.last_save_dir:
                # Use the stored directory directly
                new_file_path = os.path.join(self.last_save_dir, new_file_name)
            else:
                # Open the file dialog to select the output folder
                dialog = QFileDialog()
                output_folder = dialog.getExistingDirectory(self, 'Select Output Folder', initial_dir)

                if output_folder:
                    new_file_path = os.path.join(output_folder, new_file_name)

                    # Update the last save directory
                    self.last_save_dir = output_folder
                else:
                    return

        # Check if the file already exists
        if os.path.exists(new_file_path):
            # Show a warning message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Image already exists!")
            msg.setInformativeText("The image file already exists in the target folder. Do you want to overwrite it?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)

            # Execute the message box
            response = msg.exec_()
            if response == QMessageBox.No:
                return

        # Save the filtered image with the new file path
        cv2.imwrite(new_file_path, self.original_image)

    def updateColor(self):
        if not hasattr(self, 'original_image'):
            return

        # Get the current values from the sliders
        h_min = self.h_min_slider.value()
        s_min = self.s_min_slider.value()
        v_min = self.v_min_slider.value()

        h_max = self.h_max_slider.value()
        s_max = self.s_max_slider.value()
        v_max = self.v_max_slider.value()

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HLS)

        # Define the lower and upper bounds of HSV
        lower_bound = (h_min, s_min, v_min)
        upper_bound = (h_max, s_max, v_max)

        # Create a mask based on the lower and upper bounds
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Convert the mask to RGB color space
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Convert the RGB image to QImage and set it on the filtered image label
        height, width, channel = mask_rgb.shape
        q_image = QImage(mask_rgb.data, width, height, width * channel, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.filtered_image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        self.filtered_image = mask

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorSlider()
    # window.resize(800, 600)  # Set a specific window size if desired
    window.show()
    sys.exit(app.exec_())
