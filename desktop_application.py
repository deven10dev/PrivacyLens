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
                            QListWidget, QListWidgetItem, QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QIcon

# Import the frame extraction app
from frame_extraction import FrameExtractionApp
from face_anonymizer_images import FaceAnonymizationBatchApp

class VideoProcessingThread(QThread):
    """Thread for processing videos with deface without freezing the UI"""
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(QImage, int, int)  # current frame, current frame number, total frames
    processing_finished = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, input_file, output_file, options):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.options = options
        self.is_running = True
    
    def run(self):
        try:
            # Create output folder if it doesn't exist
            output_dir = os.path.dirname(self.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            self.log_message.emit(f"Processing video: {os.path.basename(self.input_file)}")
            
            # Build the deface command
            cmd = ["deface", str(self.input_file), "-o", str(self.output_file)]
            
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
            
            # Open video to get total frames (for progress calculation)
            try:
                cap = cv2.VideoCapture(self.input_file)
                if not cap.isOpened():
                    self.log_message.emit("Warning: Could not open input video file.")
                    total_frames = 0
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # Some video formats don't report frame count correctly
                    if total_frames <= 0:
                        self.log_message.emit("Warning: Could not determine total frames from metadata.")
                        # Try to estimate by seeking to the end
                        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                        total_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)  # Reset to beginning
                    
                    if total_frames > 0:
                        self.log_message.emit(f"Total frames in video: {total_frames}")
                    else:
                        self.log_message.emit("Warning: Unable to determine total frames. Progress will be estimated.")
                        total_frames = 0
                    
                cap.release()
            except Exception as e:
                self.log_message.emit(f"Warning: Could not determine total frames. Progress may be inaccurate. Error: {str(e)}")
                total_frames = 0
            
            # Execute deface with a pipe to capture realtime output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                # creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            
            # Set up a timer to read the output video and send frame updates
            # This allows us to show progress and preview while deface is running
            preview_cap = None
            frame_count = 0
            
            # Define a function to check if the output file exists and is readable
            def check_output_file():
                if os.path.exists(self.output_file):
                    try:
                        cap = cv2.VideoCapture(self.output_file)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            cap.release()
                            return ret
                    except:
                        pass
                return False
            
            # Wait for the output file to be created (with timeout)
            start_time = time.time()
            file_ready = False
            
            while not file_ready and time.time() - start_time < 30:  # 30 seconds timeout
                file_ready = check_output_file()
                if not file_ready:
                    time.sleep(0.5)  # Check every half second
                    self.log_message.emit("Waiting for output file to be created...")
            
            if file_ready:
                self.log_message.emit("Output file created. Starting preview...")
                
                # Set up a timer to periodically check the output file and update the preview
                preview_timer = QTimer()
                last_modified = 0
                last_position = 0
                last_heartbeat = time.time()
                
                def update_preview():
                    nonlocal last_modified, last_position, frame_count, last_heartbeat
                    
                    # Add heartbeat to show activity
                    current_time = time.time()
                    if current_time - last_heartbeat > 5:  # Every 5 seconds
                        last_heartbeat = current_time
                        self.log_message.emit(f"Still processing... (frame count: {frame_count})")
                        self.frame_processed.emit(QImage(), frame_count, total_frames)
                    
                    # Check if the file has been modified
                    try:
                        current_modified = os.path.getmtime(self.output_file)
                        
                        if current_modified > last_modified:
                            last_modified = current_modified
                            
                            # Try to open the video and read the latest frame
                            cap = cv2.VideoCapture(self.output_file)
                            
                            if cap.isOpened():
                                # Get current frame count
                                current_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                
                                # Always update frame count even if we can't get a new image
                                if current_frames > frame_count:
                                    frame_count = current_frames
                                    if total_frames > 0:
                                        progress = min(int((frame_count / total_frames) * 100), 99)
                                        self.progress_updated.emit(progress)
                                        
                                    # Send frame counter update REGARDLESS of whether we get a frame
                                    self.frame_processed.emit(QImage(), frame_count, total_frames)
                                    
                                    # Try to get the latest frame for preview (this might fail)
                                    try:
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frames - 1)
                                        ret, frame = cap.read()
                                        if ret:
                                            # Convert and emit with image
                                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            h, w, ch = rgb_frame.shape
                                            bytes_per_line = ch * w
                                            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                                            
                                            # Emit the frame for display
                                            self.frame_processed.emit(qt_image, frame_count, total_frames)
                                    except:
                                        pass  # Still sent counter update even if image fails
                                
                                cap.release()
                    except Exception as e:
                        self.log_message.emit(f"Preview update error: {str(e)}")
                
                # Initialize a counter for logging frame updates
                # We don't want to flood the log with every frame update
                log_counter = 0
                log_interval = 10  # Log every 10th frame
                
                # Use a regular timer in the thread to check the output file
                while self.is_running and process.poll() is None:
                    update_preview()
                    
                    # Increment log counter
                    log_counter += 1
                    
                    # Only log frame progress at intervals to avoid flooding the log
                    if log_counter % log_interval == 0 and frame_count > 0:
                        if total_frames > 0:
                            self.log_message.emit(f"Processing frame: {frame_count}/{total_frames} ({(frame_count/total_frames*100):.1f}%)")
                        else:
                            self.log_message.emit(f"Processing frame: {frame_count}")
                    
                    time.sleep(0.5)  # Check every half second
            
            # Process is still running, wait for it to complete
            stdout, stderr = process.communicate()
            
            # Check if successful
            if process.returncode == 0:
                self.log_message.emit("Video processing completed successfully")
                self.progress_updated.emit(100)  # Ensure 100% at the end
                self.processing_finished.emit("Video processing completed")
            else:
                error_msg = f"Error processing video: {stderr}"
                self.log_message.emit(error_msg)
                
                # Check for common errors and provide more helpful feedback
                if "moov atom not found" in stderr:
                    self.log_message.emit("This error often occurs with corrupted MP4 files or files with metadata issues.")
                    self.log_message.emit("Possible solutions:")
                    self.log_message.emit("1. Try re-encoding the video with ffmpeg: ffmpeg -i input.mp4 -c copy fixed.mp4")
                    self.log_message.emit("2. Use a different video format like .avi or .mkv")
                    self.log_message.emit("3. If the video plays in a media player, try capturing it with a screen recorder")
                    self.processing_finished.emit("Processing failed - MP4 metadata issue detected")
                elif "Failed to create detector" in stderr:
                    self.log_message.emit("This error may occur if there are issues with the face detection model.")
                    self.log_message.emit("Try reinstalling deface: pip uninstall deface && pip install deface")
                    self.processing_finished.emit("Processing failed - Detection model issue")
                else:
                    self.processing_finished.emit("Processing failed")
            
        except Exception as e:
            error_msg = f"Error during video processing: {str(e)}"
            self.log_message.emit(error_msg)
            self.processing_finished.emit(error_msg)
    
    def stop(self):
        """Stop the processing"""
        self.is_running = False

class WelcomeScreen(QWidget):
    """Welcome screen widget that appears when the app launches"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Add some spacing at the top
        layout.addSpacing(40)
        
        # Title label
        title_label = QLabel("Face Anonymization Tool")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Powered by deface library")
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)
        
        layout.addSpacing(40)
        
        # Instructions
        instructions = QLabel(
            "This tool allows you to anonymize faces in videos and images.\n"
            "You can process multiple videos in batch mode and customize anonymization options."
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addSpacing(30)
        
        # Button styles
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        
        # Get started button
        self.start_button = QPushButton("Video Face Anonymization")
        self.start_button.setMinimumSize(300, 50)
        self.start_button.setStyleSheet(button_style)
        
        # Frame extraction button
        self.extract_frames_button = QPushButton("Extract Frames from Videos")
        self.extract_frames_button.setMinimumSize(300, 50)
        self.extract_frames_button.setStyleSheet(button_style)
        
        # Image anonymization button
        self.image_anon_button = QPushButton("Image Face Anonymization")
        self.image_anon_button.setMinimumSize(300, 50)
        self.image_anon_button.setStyleSheet(button_style)

        # Update the buttons layout to include the new button
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(20)

        button_container_layout1 = QHBoxLayout()
        button_container_layout1.addStretch()
        button_container_layout1.addWidget(self.start_button)
        button_container_layout1.addStretch()

        button_container_layout2 = QHBoxLayout()
        button_container_layout2.addStretch()
        button_container_layout2.addWidget(self.extract_frames_button)
        button_container_layout2.addStretch()

        button_container_layout3 = QHBoxLayout()
        button_container_layout3.addStretch()
        button_container_layout3.addWidget(self.image_anon_button)
        button_container_layout3.addStretch()

        buttons_layout.addLayout(button_container_layout1)
        buttons_layout.addLayout(button_container_layout2)
        buttons_layout.addLayout(button_container_layout3)

        layout.addLayout(buttons_layout)
        
        # Version info at bottom
        layout.addStretch()
        version_label = QLabel("ACCESS Video Anonymization Tool v1.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)
        
        self.setLayout(layout)

class FaceAnonymizationVideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Anonymization Tools")
        self.setMinimumSize(800, 600)
        
        self.frame_extraction_window = None
        self.image_anonymization_window = None
        self.video_anonymization_window = None
        
        # Create welcome screen
        self.welcome_screen = WelcomeScreen()
        self.welcome_screen.start_button.clicked.connect(self.open_video_anonymization)
        self.welcome_screen.extract_frames_button.clicked.connect(self.open_frame_extraction)
        self.welcome_screen.image_anon_button.clicked.connect(self.open_image_anonymization)
        
        self.setCentralWidget(self.welcome_screen)
    
    def open_video_anonymization(self):
        """Open the video anonymization tool in a separate window"""
        if not self.video_anonymization_window or not self.video_anonymization_window.isVisible():
            self.video_anonymization_window = VideoAnonymizationApp()
            self.video_anonymization_window.show()
        else:
            self.video_anonymization_window.setWindowState(self.video_anonymization_window.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
            self.video_anonymization_window.activateWindow()
    
    def open_frame_extraction(self):
        """Open the frame extraction tool in a separate window"""
        if not self.frame_extraction_window or not self.frame_extraction_window.isVisible():
            self.frame_extraction_window = FrameExtractionApp()
            self.frame_extraction_window.show()
        else:
            self.frame_extraction_window.setWindowState(self.frame_extraction_window.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
            self.frame_extraction_window.activateWindow()
    
    def open_image_anonymization(self):
        """Open the image face anonymization tool in a separate window"""
        if not self.image_anonymization_window or not self.image_anonymization_window.isVisible():
            self.image_anonymization_window = FaceAnonymizationBatchApp()
            self.image_anonymization_window.show()
        else:
            self.image_anonymization_window.setWindowState(self.image_anonymization_window.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
            self.image_anonymization_window.activateWindow()
            
# Add this class to separate video functionality
class VideoAnonymizationApp(QMainWindow):
    """Standalone window for video anonymization"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Face Anonymization (powered by deface)")
        self.setMinimumSize(1000, 700)
        
        self.input_file = ""
        self.output_file = ""
        self.is_processing = False
        self.processing_thread = None
        
        # Check if deface is installed
        try:
            result = subprocess.run(
                ["deface", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=True
            )
            self.deface_version = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            QMessageBox.critical(
                self,
                "Deface Not Found",
                "The deface library was not found. Please install it using:\n\npython -m pip install deface"
            )
            return
        
        self.has_ffmpeg = False
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                self.has_ffmpeg = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        # Create the main widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Initialize UI
        self.init_ui()
        
        # Log application startup
        self.append_log(f"Video Face Anonymization App started (powered by deface {self.deface_version})")
        self.append_log("Ready to process videos")
    
    # Copy all the relevant UI and processing methods from FaceAnonymizationVideoApp
    # But remove the welcome screen related code
    def init_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout()

        # Set main layout
        self.main_widget.setLayout(main_layout)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input video:")
        self.input_path_label = QLabel("No file selected")
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path_label, 1)
        
        # Add a button for multiple file selection
        self.browse_multiple_btn = QPushButton("Add Multiple Videos")
        self.browse_multiple_btn.clicked.connect(self.browse_multiple_files)
        input_layout.addWidget(self.browse_multiple_btn)
        
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output folder:")
        self.output_path_label = QLabel("No folder selected")
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output_folder)  # Changed to select folder
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label, 1)
        output_layout.addWidget(self.browse_output_btn)
        
        # Add a batch processing list widget
        batch_layout = QVBoxLayout()
        batch_layout.addWidget(QLabel("Batch Processing Queue:"))
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.batch_list.setMinimumHeight(100)
        batch_layout.addWidget(self.batch_list)
        
        # Add buttons to manage the batch list
        batch_buttons_layout = QHBoxLayout()
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self.remove_selected_videos)
        self.clear_batch_btn = QPushButton("Clear All")
        self.clear_batch_btn.clicked.connect(self.clear_batch)
        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_item_up)
        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_item_down)
        
        batch_buttons_layout.addWidget(self.remove_selected_btn)
        batch_buttons_layout.addWidget(self.clear_batch_btn)
        batch_buttons_layout.addWidget(self.move_up_btn)
        batch_buttons_layout.addWidget(self.move_down_btn)
        batch_layout.addLayout(batch_buttons_layout)
        
        file_layout.addLayout(input_layout)
        file_layout.addLayout(output_layout)
        file_layout.addLayout(batch_layout)
        file_group.setLayout(file_layout)
        
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
        
        # Preview
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("background-color: #f0f0f0;")
        self.preview_label.setText("Video preview will appear here during processing")
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.status_label = QLabel("Ready")
        self.frame_counter_label = QLabel("Frame: 0/0")
        self.frame_counter_label.setMinimumWidth(150)  # Ensure enough space for frame counts
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.frame_counter_label)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        # Add a batch process button
        self.batch_process_btn = QPushButton("Process All Videos")
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        self.batch_process_btn.setEnabled(False)
        buttons_layout.addWidget(self.batch_process_btn)

        self.about_btn = QPushButton("About deface")
        self.about_btn.clicked.connect(self.show_about)
        
        self.force_stop_btn = QPushButton("Force Stop")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.clicked.connect(self.stop_processing)
        self.force_stop_btn.setStyleSheet("background-color: #ff6666;")
        
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
        main_layout.addWidget(file_group)
        main_layout.addWidget(options_group)
        main_layout.addWidget(preview_group)
        main_layout.addWidget(log_group)
        main_layout.addLayout(progress_layout)
        main_layout.addLayout(buttons_layout)
        
        # Set main layout
        self.main_widget.setLayout(main_layout)
        
        # Log application startup
        self.append_log(f"Video Face Anonymization App started (powered by deface {self.deface_version})")
        self.append_log("Ready to process videos")

    def open_image_anonymization(self):
        """Open the image face anonymization tool in a separate window"""
        # Create a new instance of the image anonymization app if it doesn't exist or was closed
        if not self.image_anonymization_window or not self.image_anonymization_window.isVisible():
            self.image_anonymization_window = FaceAnonymizationBatchApp()
            self.image_anonymization_window.show()
        else:
            # If window exists, bring it to front
            self.image_anonymization_window.setWindowState(self.image_anonymization_window.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
            self.image_anonymization_window.activateWindow()
            
        self.append_log("Opened Image Face Anonymization Tool")
        
    def open_frame_extraction(self):
        """Open the frame extraction tool in a separate window"""
        # Create a new instance of the frame extraction app if it doesn't exist or was closed
        if not self.frame_extraction_window or not self.frame_extraction_window.isVisible():
            self.frame_extraction_window = FrameExtractionApp()
            self.frame_extraction_window.show()
        else:
            # If window exists, bring it to front
            self.frame_extraction_window.setWindowState(self.frame_extraction_window.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
            self.frame_extraction_window.activateWindow()
            
        self.append_log("Opened Frame Extraction Tool")

    def browse_multiple_files(self):
        """Open file dialog to select multiple input video files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Input Videos", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*.*)"
        )
        if file_paths:
            # Add the files to the batch processing list
            for file_path in file_paths:
                self.add_to_batch(file_path)
            
            self.append_log(f"Added {len(file_paths)} videos to batch processing queue")
            self.update_batch_process_button()
    
    def add_to_batch(self, file_path):
        """Add a file to the batch processing list"""
        # Check if the file is already in the list
        for i in range(self.batch_list.count()):
            item = self.batch_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == file_path:
                # Already exists
                return
        
        # Add the file to the list
        item = QListWidgetItem(os.path.basename(file_path))
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        self.batch_list.addItem(item)
    
    def remove_selected_videos(self):
        """Remove selected videos from the batch list"""
        selected_items = self.batch_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.batch_list.row(item)
            self.batch_list.takeItem(row)
        
        self.append_log(f"Removed {len(selected_items)} video(s) from batch queue")
        self.update_batch_process_button()
    
    def clear_batch(self):
        """Clear all videos from the batch list"""
        count = self.batch_list.count()
        if count > 0:
            self.batch_list.clear()
            self.append_log(f"Cleared all {count} videos from batch queue")
            self.update_batch_process_button()
    
    def move_item_up(self):
        """Move the selected item up in the batch list"""
        current_row = self.batch_list.currentRow()
        if current_row > 0:
            item = self.batch_list.takeItem(current_row)
            self.batch_list.insertItem(current_row - 1, item)
            self.batch_list.setCurrentItem(item)
    
    def move_item_down(self):
        """Move the selected item down in the batch list"""
        current_row = self.batch_list.currentRow()
        if current_row < self.batch_list.count() - 1:
            item = self.batch_list.takeItem(current_row)
            self.batch_list.insertItem(current_row + 1, item)
            self.batch_list.setCurrentItem(item)
    
    def browse_output_folder(self):
        """Open folder dialog to select output directory"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder_path:
            self.output_file = folder_path
            self.output_path_label.setText(os.path.basename(folder_path))
            self.append_log(f"Output folder set to: {folder_path}")
            self.update_batch_process_button()
    
    def update_batch_process_button(self):
        """Update the batch process button state based on list and output folder"""
        has_videos = self.batch_list.count() > 0
        has_output = bool(self.output_file)
        
        self.batch_process_btn.setEnabled(has_videos and has_output)
    
    def start_batch_processing(self):
        """Start processing all videos in the batch list"""
        if self.is_processing:
            self.stop_processing()
            return
        
        if self.batch_list.count() == 0:
            QMessageBox.warning(self, "No Videos", "No videos in the batch processing queue.")
            return
        
        if not self.output_file:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder.")
            return
        
        # Start batch processing
        self.current_batch_index = 0
        self.is_processing = True
        self.batch_process_btn.setText("Stop Batch Processing")
        
        # Disable UI elements during batch processing
        self.disable_ui_during_processing(True)
        
        # Start processing the first video
        self.process_next_batch_video()
    
    def process_next_batch_video(self):
        """Process the next video in the batch queue"""
        if not self.is_processing or self.current_batch_index >= self.batch_list.count():
            # We're done or stopped
            self.batch_processing_complete()
            return
        
        # Get the next video from the queue
        item = self.batch_list.item(self.current_batch_index)
        input_file = item.data(Qt.ItemDataRole.UserRole)
        
        # Set the item background to indicate it's being processed
        item.setBackground(QColor(255, 255, 200))  # Light yellow
        self.batch_list.scrollToItem(item)
        
        # Generate output filename
        input_name = os.path.basename(input_file)
        input_base = os.path.splitext(input_name)[0]
        input_ext = os.path.splitext(input_name)[1]
        output_file = os.path.join(
            self.output_file,
            f"{input_base}_anonymized{input_ext}"
        )
        
        # Set as current files
        self.input_file = input_file
        self.output_file_current = output_file  # Store current output file separately
        
        # Update labels
        self.input_path_label.setText(os.path.basename(input_file))
        self.status_label.setText(f"Processing video {self.current_batch_index + 1} of {self.batch_list.count()}")
        
        # Log
        self.append_log(f"Batch processing: Starting video {self.current_batch_index + 1} of {self.batch_list.count()}")
        self.append_log(f"Input: {input_file}")
        self.append_log(f"Output: {output_file}")
        
        # Show thumbnail
        self.show_video_thumbnail(input_file)
        
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
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        self.frame_counter_label.setText("Frame: 0/0")
        
        # Create and start the processing thread
        self.processing_thread = VideoProcessingThread(
            input_file,
            output_file,
            options
        )
        
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.frame_processed.connect(self.update_frame_preview)
        self.processing_thread.processing_finished.connect(self.batch_video_finished)
        self.processing_thread.log_message.connect(self.append_log)
        
        # Start processing
        self.processing_thread.start()
        
        # Enable force stop button
        self.force_stop_btn.setEnabled(True)
    
    def batch_video_finished(self, message):
        """Handle when a video in the batch is finished"""
        # Mark the current video as done
        if 0 <= self.current_batch_index < self.batch_list.count():
            item = self.batch_list.item(self.current_batch_index)
            if "completed" in message.lower():
                # Success
                item.setBackground(QColor(200, 255, 200))  # Light green
                self.append_log(f"Successfully processed: {os.path.basename(item.data(Qt.ItemDataRole.UserRole))}")
            else:
                # Error
                item.setBackground(QColor(255, 200, 200))  # Light red
                self.append_log(f"Failed to process: {os.path.basename(item.data(Qt.ItemDataRole.UserRole))}")
        
        # Move to the next video
        self.current_batch_index += 1
        
        if self.is_processing and self.current_batch_index < self.batch_list.count():
            # Process the next video
            QTimer.singleShot(1000, self.process_next_batch_video)  # Add a small delay between videos
        else:
            # Batch processing complete
            self.batch_processing_complete()
    
    def batch_processing_complete(self):
        """Handle completion of batch processing"""
        self.is_processing = False
        self.batch_process_btn.setText("Process All Videos")
        self.force_stop_btn.setEnabled(False)
        
        # Re-enable UI
        self.disable_ui_during_processing(False)
        
        # Display results
        if self.current_batch_index > 0:
            processed_count = self.current_batch_index
            total_count = self.batch_list.count()
            remaining = total_count - processed_count
            
            status_msg = f"Batch processing complete. Processed {processed_count} of {total_count} videos."
            self.append_log(status_msg)
            self.status_label.setText("Batch processing complete")
            
            QMessageBox.information(
                self,
                "Batch Processing Complete",
                f"{status_msg}\n\nOutput folder: {self.output_file}"
            )
        
        # Reset batch index
        self.current_batch_index = -1
        
    def processing_finished(self, message):
        """Handle the end of processing for single video mode"""
        self.is_processing = False
        self.status_label.setText(message)
        
        # Disable force stop button
        self.force_stop_btn.setEnabled(False)
        
        # Re-enable UI elements
        self.disable_ui_during_processing(False)
        
        # Show a message box if completed successfully
        if "completed" in message.lower():
            QMessageBox.information(
                self,
                "Processing Complete",
                f"{message}\n\nOutput file: {self.output_file}"
            )

    def stop_processing(self):
        """Stop the video processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.status_label.setText("Stopping processing...")
            self.append_log("Stopping processing - please wait...")
            
        # If in batch mode, reset batch state
        if self.current_batch_index >= 0:
            self.append_log("Batch processing stopped by user")
            self.batch_processing_complete()

    def disable_ui_during_processing(self, disable):
        """Enable/disable UI elements during processing"""
        self.browse_output_btn.setEnabled(not disable)
        self.browse_multiple_btn.setEnabled(not disable)
        self.anon_method.setEnabled(not disable)
        self.mosaic_size.setEnabled(not disable)
        self.threshold_slider.setEnabled(not disable)
        self.mask_scale_slider.setEnabled(not disable)
        self.scale_combo.setEnabled(not disable)
        self.box_check.setEnabled(not disable)
        self.draw_scores_check.setEnabled(not disable)

        # Disable batch control buttons as well
        self.remove_selected_btn.setEnabled(not disable)
        self.clear_batch_btn.setEnabled(not disable)
        self.move_up_btn.setEnabled(not disable)
        self.move_down_btn.setEnabled(not disable)
        self.batch_list.setEnabled(not disable)

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
    
    def show_video_thumbnail(self, video_path):
        """Show a thumbnail of the first frame of the video"""
        try:
            # Try using ffmpeg to get a thumbnail if available (to handle corrupt headers)
            try:
                # First attempt: try to use ffmpeg directly if available
                thumbnail_path = os.path.join(os.path.dirname(video_path), f"temp_thumbnail_{int(time.time())}.jpg")
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-ss", "00:00:01", "-vframes", "1", thumbnail_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                
                if os.path.exists(thumbnail_path) and os.path.getsize(thumbnail_path) > 0:
                    # Load the thumbnail
                    frame = cv2.imread(thumbnail_path)
                    if frame is not None:
                        # Convert to RGB and display
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = frame.shape
                        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                        scaled_pixmap = pixmap.scaled(
                            self.preview_label.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self.preview_label.setPixmap(scaled_pixmap)
                        
                        # Clean up
                        try:
                            os.remove(thumbnail_path)
                        except:
                            pass
                            
                        # Get video info using ffprobe
                        try:
                            result = subprocess.run(
                                ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                                 "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", 
                                 "-of", "csv=p=0", video_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=5
                            )
                            
                            if result.returncode == 0 and result.stdout:
                                info = result.stdout.strip().split(',')
                                if len(info) >= 4:
                                    width, height, fps_str, frame_count = info
                                    
                                    # Parse FPS (can be in form "30000/1001")
                                    try:
                                        if '/' in fps_str:
                                            num, den = map(float, fps_str.split('/'))
                                            fps = num / den if den else 0
                                        else:
                                            fps = float(fps_str)
                                    except:
                                        fps = 0
                                        
                                    # Log video properties
                                    duration_sec = int(frame_count) / fps if fps > 0 and frame_count and frame_count != 'N/A' else 0
                                    duration_str = time.strftime('%H:%M:%S', time.gmtime(duration_sec))
                                    
                                    self.append_log(f"Video properties (from ffprobe):")
                                    self.append_log(f"  Resolution: {width}x{height}")
                                    self.append_log(f"  FPS: {fps:.2f}")
                                    
                                    if frame_count and frame_count != 'N/A':
                                        self.append_log(f"  Duration: {duration_str} ({frame_count} frames)")
                                    else:
                                        self.append_log(f"  Duration: Unknown (frame count not available)")
                                        
                                    return  # Success with ffmpeg/ffprobe
                        except Exception as e:
                            self.append_log(f"Error getting video info with ffprobe: {str(e)}")
                
            except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
                self.append_log(f"Could not use ffmpeg for thumbnail: {str(e)}")
            
            # Fallback to OpenCV if ffmpeg failed
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.append_log("Warning: OpenCV could not open the video file. The file may be corrupted or use an unsupported codec.")
                self.preview_label.setText("Could not load video preview\nTry a different video format or check if the file is corrupted")
                return
                
            ret, frame = cap.read()
            if not ret:
                self.append_log("Warning: Could not read the first frame of the video.")
                self.preview_label.setText("Could not read video frame\nThe file may be corrupted")
                cap.release()
                return
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Log video properties
            duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0
            duration_str = time.strftime('%H:%M:%S', time.gmtime(duration_sec))
            
            self.append_log(f"Video properties (from OpenCV):")
            self.append_log(f"  Resolution: {width}x{height}")
            self.append_log(f"  FPS: {fps}")
            
            if frame_count > 0:
                self.append_log(f"  Duration: {duration_str} ({frame_count} frames)")
            else:
                self.append_log(f"  Duration: Unknown (frame count not available)")
            
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            
            # Create QImage and QPixmap
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale the pixmap to fit the preview label
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
            cap.release()
            
        except Exception as e:
            self.append_log(f"Error showing video thumbnail: {str(e)}")
            self.preview_label.setText("Could not load video preview\nThe file may be corrupted or in an unsupported format")
       
    def toggle_processing(self):
        """Start or stop video processing"""
        if not self.is_processing:
            self.start_processing()
        else:
            self.stop_processing()

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
        
        # Create and start the processing thread
        self.processing_thread = VideoProcessingThread(
            self.input_file,
            self.output_file,
            options
        )
        
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.frame_processed.connect(self.update_frame_preview)
        self.processing_thread.processing_finished.connect(self.processing_finished)
        self.processing_thread.log_message.connect(self.append_log)
        
        # Start processing
        self.processing_thread.start()
        
        # Update status
        self.status_label.setText("Processing...")
        
        # Enable force stop button
        self.force_stop_btn.setEnabled(True)
        
        # Disable UI elements during processing
        self.disable_ui_during_processing(True)
    
    def stop_processing(self):
        """Stop the video processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.status_label.setText("Stopping processing...")
            self.append_log("Stopping processing - please wait...")
    
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}%")
    
    def update_frame_preview(self, image, current_frame=0, total_frames=0):
        """Update the preview with the current processed frame"""
        pixmap = QPixmap.fromImage(image)
        
        # Scale the pixmap to fit the preview label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.preview_label.setPixmap(scaled_pixmap)
        
        # Update the frame counter label with frame information
        if total_frames > 0:
            self.frame_counter_label.setText(f"Frame: {current_frame}/{total_frames}")
            # Also update the status text for more visibility
            self.status_label.setText(f"Processing: {(current_frame/total_frames*100):.1f}%")
        elif current_frame > 0:
            self.frame_counter_label.setText(f"Frame: {current_frame}")
            self.status_label.setText("Processing...")
        else:
            self.frame_counter_label.setText("Frame: 0/0")
            self.status_label.setText("Processing...")
      
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

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            # Check deface
            subprocess.run(["deface", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            QMessageBox.critical(
                self,
                "Deface Not Found",
                "The deface library was not found. Please install it using:\n\npip install deface"
            )
            return False
        return True

    def closeEvent(self, event):
        """Handle window close event - stop any running processes"""
        if self.is_processing and self.processing_thread and self.processing_thread.isRunning():
            self.append_log("Window closing - stopping all processing...")
            self.processing_thread.stop()
            self.processing_thread.wait(1000)  # Wait up to 1 second for graceful termination
            
            # Force termination if still running
            if self.processing_thread.isRunning():
                self.processing_thread.terminate()
            
            self.is_processing = False
        
        # Accept the close event
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = FaceAnonymizationVideoApp()
    window.show()
    sys.exit(app.exec())