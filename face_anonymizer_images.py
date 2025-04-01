import sys
import os
import subprocess
from pathlib import Path
import time
import datetime
import cv2
import numpy as np
import platform

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QWidget,
                            QProgressBar, QComboBox, QSpinBox, QCheckBox, QGroupBox,
                            QRadioButton, QButtonGroup, QMessageBox, QPlainTextEdit,
                            QListWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor


class BatchProcessingThread(QThread):
    """Thread for processing multiple images with deface without freezing the UI"""
    progress_updated = pyqtSignal(int)
    image_processed = pyqtSignal(str, QImage)  # filename, image
    processing_finished = pyqtSignal(str)
    log_message = pyqtSignal(str)
    current_file_changed = pyqtSignal(str)
    
    def __init__(self, input_folders, output_folder, options, use_custom_output):
        super().__init__()
        self.input_folders = input_folders  # Changed to accept a list of folders
        self.output_folder = output_folder
        self.use_custom_output = use_custom_output
        self.options = options
        self.is_running = True

    def run(self):
        try:
            # Process each image
            all_image_files = []
            input_to_output_folders = {}  # Map input folders to their output folders
            
            # Create mapping for each input folder to its output folder
            for input_folder in self.input_folders:
                if self.use_custom_output and self.output_folder and os.path.exists(self.output_folder):
                    # If user specified a custom output folder, create subfolders inside it
                    input_folder_name = os.path.basename(input_folder)
                    output_folder = os.path.join(
                        self.output_folder,
                        f"{input_folder_name}_anonymized"
                    )
                else:
                    # Default behavior - create output folder next to input folder
                    output_folder = os.path.join(
                        os.path.dirname(input_folder),
                        f"{os.path.basename(input_folder)}_anonymized"
                    )
                
                input_to_output_folders[input_folder] = output_folder
                os.makedirs(output_folder, exist_ok=True)
                self.log_message.emit(f"Created output folder: {output_folder}")

            # Collect all image files from input folders
            for input_folder in self.input_folders:
                # Get list of image files
                image_files = []
                for ext in ('*.jpg', '*.jpeg', '*.png'):
                    image_files.extend(Path(input_folder).glob(ext))
                    
                # Filter out non-image files and sort them
                image_files = sorted([f for f in image_files if f.is_file()])
                
                # Track which input folder each file belongs to
                for f in image_files:
                    all_image_files.append((f, input_folder))
            
            if not all_image_files:
                self.log_message.emit("No image files found in the selected folders.")
                self.processing_finished.emit("No images to process")
                return
            
            total_files = len(all_image_files)
            self.log_message.emit(f"Found {total_files} images to process")
            
            # Process each image
            for i, (image_path, input_folder) in enumerate(all_image_files):
                if not self.is_running:
                    self.log_message.emit("Processing stopped by user")
                    self.processing_finished.emit("Processing stopped by user")
                    return
                
                self.current_file_changed.emit(str(image_path.name))
                self.log_message.emit(f"Processing image {i+1}/{total_files}: {image_path.name}")
                
                # Get the output folder for this input folder
                output_folder = input_to_output_folders[input_folder]
                output_path = Path(output_folder) / f"{image_path.stem}_anonymized{image_path.suffix}"
                
                # Build the deface command
                cmd = ["deface", str(image_path), "-o", str(output_path)]
                                
                # Add options based on user selections
                if self.options["threshold"] != 0.2:  # Default is 0.2
                    cmd.extend(["--thresh", str(self.options["threshold"])])
                
                if self.options["mask_scale"] != 1.3:  # Default is 1.3
                    cmd.extend(["--mask-scale", str(self.options["mask_scale"])])
                
                if self.options["anonymization_method"] != "blur":
                    cmd.extend(["--replacewith", self.options["anonymization_method"]])
                
                if self.options["anonymization_method"] == "mosaic" and self.options["mosaic_size"] != 20:
                    cmd.extend(["--mosaicsize", str(self.options["mosaic_size"])])
                
                if self.options["box_method"]:
                    cmd.append("--boxes")
                
                if self.options["draw_scores"]:
                    cmd.append("--draw-scores")
                
                # Scaling option for detection
                if self.options["scale"]:
                    cmd.extend(["--scale", self.options["scale"]])
                
                # Log the command being executed
                cmd_str = " ".join(cmd)
                self.log_message.emit(f"Executing command: {cmd_str}")
                
                # Execute the deface command
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True,
                    # creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
                
                stdout, stderr = process.communicate()
                
                # Log output
                if stdout:
                    self.log_message.emit(stdout.strip())
                
                # Check if successful
                if process.returncode == 0:
                    self.log_message.emit(f"Successfully processed: {image_path.name}")
                    
                    # Load the output image for preview
                    if os.path.exists(output_path):
                        try:
                            img = cv2.imread(str(output_path))
                            if img is not None:
                                # Convert to RGB
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                h, w, ch = img.shape
                                qt_image = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
                                self.image_processed.emit(str(output_path), qt_image)
                        except Exception as e:
                            self.log_message.emit(f"Error loading processed image: {str(e)}")
                else:
                    self.log_message.emit(f"Error processing {image_path.name}: {stderr}")
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
            
            self.log_message.emit(f"Batch processing completed. Processed {total_files} images.")
            self.processing_finished.emit(f"Completed processing {total_files} images")
            
        except Exception as e:
            error_msg = f"Error during batch processing: {str(e)}"
            self.log_message.emit(error_msg)
            self.processing_finished.emit(error_msg)
    
    def stop(self):
        """Stop the processing"""
        self.is_running = False

    def stop_safely(self):
        """Safely stop processing and clean up resources"""
        self.is_running = False
        
        # Give processes time to terminate gracefully
        start_time = time.time()
        while self.isRunning() and time.time() - start_time < 2:
            time.sleep(0.1)


class FaceAnonymizationBatchApp(QMainWindow):
    """Main application window for batch processing images using deface library"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch Image Face Anonymization (powered by deface)")
        self.setMinimumSize(1000, 700)
        
        self.input_folders = []  # Changed to a list to hold multiple input folders
        self.output_folder = ""
        self.use_custom_output = False  # Add this flag to track output mode
        self.is_processing = False
        self.processing_thread = None
        self.current_preview_file = None
        
        # Check if deface is installed
        try:
            # Check if deface is installed and get version
            result = subprocess.run(
                ["deface", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=True
            )
            # Store the version for later display
            self.deface_version = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            QMessageBox.critical(
                self,
                "Deface Not Found",
                "The deface library was not found. Please install it using:\n\npython -m pip install deface"
            )
            sys.exit(1)
        
        self.init_ui()
    
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Folder selection
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout()
    
        # In the init_ui method, modify the input_layout:
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input folders:")
        self.input_path_label = QLabel("No folders selected")
        self.browse_input_btn = QPushButton("Add Folder")  # Changed label
        self.browse_input_btn.clicked.connect(self.browse_input_folders)
        self.clear_input_btn = QPushButton("Clear All")
        self.clear_input_btn.clicked.connect(self.clear_input_folders)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path_label, 1)
        input_layout.addWidget(self.browse_input_btn)
        input_layout.addWidget(self.clear_input_btn)
        
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output folder:")
        self.output_path_label = QLabel("Default (next to input folders)")
        self.output_mode_btn = QPushButton("Use Custom Output")
        self.output_mode_btn.setCheckable(True)
        self.output_mode_btn.clicked.connect(self.toggle_output_mode)
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output_folder)
        self.browse_output_btn.setEnabled(False)  # Disabled by default
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label, 1)
        output_layout.addWidget(self.output_mode_btn)
        output_layout.addWidget(self.browse_output_btn)
                
        folder_layout.addLayout(input_layout)
        folder_layout.addLayout(output_layout)
        folder_group.setLayout(folder_layout)
        
        # Processing options
        options_group = QGroupBox("Anonymization Options")
        options_layout = QVBoxLayout()
        
        # Anonymization method
        anon_layout = QHBoxLayout()
        anon_layout.addWidget(QLabel("Anonymization Method:"))
        self.anon_method = QComboBox()
        self.anon_method.addItems(["blur", "solid", "mosaic", "none"])
        self.anon_method.currentTextChanged.connect(self.update_ui_based_on_method)
        anon_layout.addWidget(self.anon_method)
        
        # Mosaic size (only visible when mosaic method selected)
        self.mosaic_layout = QHBoxLayout()
        self.mosaic_layout.addWidget(QLabel("Mosaic Size:"))
        self.mosaic_size = QSpinBox()
        self.mosaic_size.setMinimum(5)
        self.mosaic_size.setMaximum(100)
        self.mosaic_size.setValue(20)
        self.mosaic_layout.addWidget(self.mosaic_size)
        
        # Detection threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Detection Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(20)  # Default 0.2
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_value_label = QLabel("0.2")
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)
        thresh_layout.addWidget(self.threshold_slider)
        thresh_layout.addWidget(self.threshold_value_label)
        
        # Mask scale
        mask_layout = QHBoxLayout()
        mask_layout.addWidget(QLabel("Mask Scale:"))
        self.mask_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_scale_slider.setMinimum(10)
        self.mask_scale_slider.setMaximum(30)
        self.mask_scale_slider.setValue(13)  # Default 1.3
        self.mask_scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.mask_scale_slider.setTickInterval(5)
        self.mask_scale_value_label = QLabel("1.3")
        self.mask_scale_slider.valueChanged.connect(self.update_mask_scale_value)
        mask_layout.addWidget(self.mask_scale_slider)
        mask_layout.addWidget(self.mask_scale_value_label)
        
        # Downscaling for performance
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Downscale for Detection:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["None", "640x360", "1280x720", "1920x1080"])
        scale_layout.addWidget(self.scale_combo)
        
        # Checkboxes for options
        checks_layout = QVBoxLayout()
        
        self.box_check = QCheckBox("Use boxes instead of ellipse masks")
        checks_layout.addWidget(self.box_check)
        
        self.draw_scores_check = QCheckBox("Draw detection scores")
        checks_layout.addWidget(self.draw_scores_check)
        
        # Add all layouts to options
        options_layout.addLayout(anon_layout)
        options_layout.addLayout(self.mosaic_layout)
        options_layout.addLayout(thresh_layout)
        options_layout.addLayout(mask_layout)
        options_layout.addLayout(scale_layout)
        options_layout.addLayout(checks_layout)
        options_group.setLayout(options_layout)
        
        # Update UI to hide mosaic settings initially
        self.update_ui_based_on_method("blur")
        
        # Create a horizontal layout for files and preview
        files_preview_layout = QHBoxLayout()
        
        # File list
        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout()
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.load_selected_file_preview)
        files_layout.addWidget(self.file_list)
        files_group.setLayout(files_layout)
        
        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.current_file_label = QLabel("No file selected")
        preview_layout.addWidget(self.current_file_label)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet("background-color: #f0f0f0;")
        self.preview_label.setText("Image preview will appear here")
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        # Add files and preview to the horizontal layout
        files_preview_layout.addWidget(files_group, 1)
        files_preview_layout.addWidget(preview_group, 2)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        self.process_btn = QPushButton("Process Folder")
        self.process_btn.clicked.connect(self.toggle_processing)
        self.process_btn.setEnabled(False)
        
        self.about_btn = QPushButton("About deface")
        self.about_btn.clicked.connect(self.show_about)
        
        self.force_stop_btn = QPushButton("Force Stop")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.clicked.connect(self.stop_processing)
        self.force_stop_btn.setStyleSheet("background-color: #ff6666;")
        
        buttons_layout.addWidget(self.process_btn)
        buttons_layout.addWidget(self.force_stop_btn)
        buttons_layout.addWidget(self.about_btn)
        
        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)  # Limit to prevent memory issues
        self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)
        log_group.setLayout(log_layout)
        
        # Add all components to main layout
        main_layout.addWidget(folder_group)
        main_layout.addWidget(options_group)
        main_layout.addLayout(files_preview_layout)
        main_layout.addWidget(log_group)
        main_layout.addLayout(progress_layout)
        main_layout.addLayout(buttons_layout)
        
        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Check if folders are selected to enable/disable process button
        self.check_folders_selected()
        
        # Log application startup
        self.append_log(f"Batch Image Face Anonymization App started (powered by deface {self.deface_version})")
        self.append_log("Ready to process images")

    def toggle_output_mode(self):
        """Toggle between default and custom output folder modes"""
        self.use_custom_output = self.output_mode_btn.isChecked()
        
        if self.use_custom_output:
            self.output_mode_btn.setText("Use Default Output")
            self.browse_output_btn.setEnabled(True)
            if not self.output_folder:
                self.output_path_label.setText("No custom folder selected")
        else:
            self.output_mode_btn.setText("Use Custom Output")
            self.browse_output_btn.setEnabled(False)
            self.output_folder = ""
            self.output_path_label.setText("Default (next to input folders)")
            
        self.append_log(f"Output mode changed to: {'Custom' if self.use_custom_output else 'Default'}")
    
    def update_ui_based_on_method(self, method):
        """Show/hide UI elements based on the selected anonymization method"""
        # Show mosaic size only when mosaic method is selected
        for i in range(self.mosaic_layout.count()):
            item = self.mosaic_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(method == "mosaic")
    
    def update_threshold_value(self):
        """Update the threshold value label"""
        value = self.threshold_slider.value() / 100
        self.threshold_value_label.setText(f"{value:.2f}")
    
    def update_mask_scale_value(self):
        """Update the mask scale value label"""
        value = self.mask_scale_slider.value() / 10
        self.mask_scale_value_label.setText(f"{value:.1f}")
    
    def append_log(self, message):
        """Add a message to the log with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_msg = f"[{timestamp}] {message}"
        self.log_text.appendPlainText(formatted_msg)
        
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def browse_input_folders(self):
        """Open file dialog to select multiple input folders"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", ""
        )
        
        if folder_path:
            # Add the folder to our list if it's not already there
            if folder_path not in self.input_folders:
                self.input_folders.append(folder_path)
                
                # Update the displayed folder names
                folder_names = ", ".join(
                    [os.path.basename(path) for path in self.input_folders]
                )
                self.input_path_label.setText(folder_names)
                
                self.append_log(f"Added input folder: {folder_path}")
                
                # Load image files from all selected folders
                self.load_files_from_folders(self.input_folders)
                
                self.check_folders_selected()
                
    def clear_input_folders(self):
        """Clear the list of selected input folders"""
        self.input_folders = []
        self.input_path_label.setText("No folders selected")
        self.file_list.clear()
        self.preview_label.setText("Image preview will appear here")
        self.preview_label.setPixmap(QPixmap())  # Clear any existing pixmap
        self.current_file_label.setText("No file selected")
        self.append_log("Input folders cleared")
        self.check_folders_selected()

    def browse_output_folder(self):
        """Open file dialog to select output folder location"""
        if not self.use_custom_output:
            return
            
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder_path:
            self.output_folder = folder_path
            self.output_path_label.setText(os.path.basename(folder_path) if os.path.basename(folder_path) else folder_path)
            self.append_log(f"Custom output folder set to: {folder_path}")
    
    def load_files_from_folders(self, folder_paths):
        """Load image files from the selected folders into the file list"""
        self.file_list.clear()
        all_image_files = []
        
        for folder_path in folder_paths:
            # Get list of image files
            image_files = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                image_files.extend(Path(folder_path).glob(ext))
            
            # Sort files
            image_files = sorted([f for f in image_files if f.is_file()])
            all_image_files.extend(image_files)
        
        if not all_image_files:
            self.append_log("No image files found in the selected folders")
            return
        
        # Add files to list
        for file_path in all_image_files:
            self.file_list.addItem(file_path.name)
        
        self.append_log(f"Found {len(all_image_files)} images in the selected folders")
        
        # Select first file
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
    
    def load_selected_file_preview(self, current, previous):
        """Load preview of the selected file"""
        if not current:
            return
        
        file_name = current.text()
        
        # Find the folder containing the selected file
        file_path = None
        for folder in self.input_folders:
            temp_path = os.path.join(folder, file_name)
            if os.path.exists(temp_path):
                file_path = temp_path
                break
        
        if not file_path:
            self.append_log(f"Error: Could not find {file_name} in selected input folders.")
            return
        
        self.current_file_label.setText(file_name)
        self.current_preview_file = file_path
        
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is not None:
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                
                # Create QImage and QPixmap
                qt_image = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # Scale the pixmap to fit the preview label
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.preview_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.preview_label.setText(f"Error loading preview: {str(e)}")
    
    def check_folders_selected(self):
        """Check if input folders are selected"""
        if self.input_folders:
            self.process_btn.setEnabled(True)
        else:
            self.process_btn.setEnabled(False)
    
    def toggle_processing(self):
        """Start or stop batch processing"""
        if not self.is_processing:
            self.start_processing()
        else:
            self.stop_processing()
    
    def start_processing(self):
        """Start the deface batch processing"""
        self.is_processing = True
        self.process_btn.setText("Stop Processing")
        
        # Gather options
        options = {
            "threshold": float(self.threshold_value_label.text()),
            "mask_scale": float(self.mask_scale_value_label.text()),
            "anonymization_method": self.anon_method.currentText(),
            "mosaic_size": self.mosaic_size.value(),
            "box_method": self.box_check.isChecked(),
            "draw_scores": self.draw_scores_check.isChecked(),
            "scale": self.scale_combo.currentText() if self.scale_combo.currentIndex() > 0 else ""
        }
        
        # Log the output location strategy
        if self.output_folder:
            self.append_log(f"Using custom output folder: {self.output_folder}")
        else:
            self.append_log("Using default output folders next to each input folder")
        
        # Create and start the processing thread
        self.processing_thread = BatchProcessingThread(
            self.input_folders,
            self.output_folder,
            options,
            self.use_custom_output
        )
                
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.image_processed.connect(self.update_preview)
        self.processing_thread.processing_finished.connect(self.processing_finished)
        self.processing_thread.log_message.connect(self.append_log)
        self.processing_thread.current_file_changed.connect(self.update_current_file)
        
        # Start processing
        self.processing_thread.start()
        
        # Update status
        self.status_label.setText("Processing...")
        
        # Enable force stop button
        self.force_stop_btn.setEnabled(True)
        
        # Disable UI elements during processing
        self.disable_ui_during_processing(True)
    
    def stop_processing(self):
        """Stop the batch processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.status_label.setText("Stopping processing...")
            self.append_log("Stopping processing - please wait...")
    
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
    
    def update_current_file(self, file_name):
        """Update the current file being processed"""
        self.current_file_label.setText(f"Processing: {file_name}")
    
    def update_preview(self, file_path, image):
        """Update the preview with the processed image"""
        pixmap = QPixmap.fromImage(image)
        
        # Scale the pixmap to fit the preview label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.preview_label.setPixmap(scaled_pixmap)
        self.current_file_label.setText(f"Processed: {os.path.basename(file_path)}")
    
    def processing_finished(self, message):
        """Handle the end of processing"""
        self.is_processing = False
        self.process_btn.setText("Process Folder")
        self.status_label.setText(message)
        
        # Disable force stop button
        self.force_stop_btn.setEnabled(False)
        
        # Re-enable UI elements
        self.disable_ui_during_processing(False)
        
        # Show a message box if completed successfully
        if "Completed" in message:
            QMessageBox.information(
                self,
                "Processing Complete",
                f"{message}\n\nOutput folder: {self.output_folder}"
            )
    
    def disable_ui_during_processing(self, disable):
        """Enable/disable UI elements during processing"""
        self.browse_input_btn.setEnabled(not disable)
        self.browse_output_btn.setEnabled(not disable)
        self.anon_method.setEnabled(not disable)
        self.mosaic_size.setEnabled(not disable)
        self.threshold_slider.setEnabled(not disable)
        self.mask_scale_slider.setEnabled(not disable)
        self.scale_combo.setEnabled(not disable)
        self.box_check.setEnabled(not disable)
        self.draw_scores_check.setEnabled(not disable)
        self.file_list.setEnabled(not disable)
    
    def show_about(self):
        """Show information about deface"""
        about_text = (
            "deface: Image and video anonymization by face detection\n\n"
            "deface is a command-line tool for automatic anonymization of faces in images or videos. "
            "It works by detecting faces and applying an anonymization filter.\n\n"
            "Features:\n"
            "- Multiple anonymization methods (blur, solid boxes, mosaic)\n"
            "- Adjustable detection threshold\n"
            "- Support for downscaling to improve performance\n"
            "- Optional box/ellipse masking\n\n"
            "For more information, visit: https://github.com/ORB-HD/deface"
        )
        
        QMessageBox.information(self, "About deface", about_text)
    
    def closeEvent(self, event):
        """Handle window close event - stop any running processes"""
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
            # 1. Add confirmation dialog before attempting to stop
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Image processing is still running. Do you want to stop processing and close the window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            # 4. Give user option to cancel window close
            if reply == QMessageBox.StandardButton.No or reply == QMessageBox.StandardButton.Cancel:
                self.append_log("Window close canceled by user")
                event.ignore()
                return
            
            try:
                self.append_log("Window closing - stopping all image processing...")
                
                # 2. Properly disconnect signals with better exception handling
                signals_to_disconnect = [
                    (self.processing_thread.progress_updated, self.update_progress),
                    (self.processing_thread.image_processed, self.update_preview),
                    (self.processing_thread.processing_finished, self.processing_finished),
                    (self.processing_thread.log_message, self.append_log),
                    (self.processing_thread.current_file_changed, self.update_current_file)
                ]
                
                for signal, slot in signals_to_disconnect:
                    try:
                        signal.disconnect(slot)
                    except (TypeError, RuntimeError) as e:
                        # Handle case where signal was not connected
                        self.append_log(f"Note: Signal disconnect issue: {str(e)}")
                
                # Stop the thread safely
                self.processing_thread.stop_safely()  # Use stop_safely instead of stop
                self.processing_thread.wait(1500)  # Wait a bit longer
                
                # If still running after graceful attempt, confirm force quit
                if self.processing_thread.isRunning():
                    force_reply = QMessageBox.question(
                        self, "Processing Not Responding",
                        "Image processing is still running and not responding. Force quit?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if force_reply == QMessageBox.StandardButton.Yes:
                        self.append_log("Forcing thread termination")
                        self.processing_thread.terminate()
                        # Wait a brief moment for cleanup
                        self.processing_thread.wait(300)
                    else:
                        self.append_log("Force quit canceled - window will remain open")
                        event.ignore()
                        return
            # 3. Better exception handling
            except Exception as e:
                error_message = f"Error during thread cleanup: {str(e)}"
                self.append_log(f"ERROR: {error_message}")
                
                # Show error to user
                QMessageBox.warning(
                    self,
                    "Error During Shutdown",
                    f"An error occurred while trying to shut down:\n\n{error_message}\n\nThe application will close."
                )
        
        # Log final closure
        if hasattr(self, 'append_log'):  # Check if method exists before window is fully initialized
            self.append_log("Application closing")
            
        # Accept the close event
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceAnonymizationBatchApp()
    window.show()
    sys.exit(app.exec())
