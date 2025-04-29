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
from face_anonymizer_videos import FaceAnonymizationVideoApp

# Import deface module directly
from centerface import CenterFace
import deface

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
            
            # Configure options for deface module
            threshold = self.options["threshold"]
            mask_scale = self.options["mask_scale"]
            replacewith = self.options["anonymization_method"]
            ellipse = not self.options["box_method"]
            draw_scores = self.options["draw_scores"]
            mosaicsize = self.options["mosaic_size"] if "mosaic_size" in self.options else 20
            
            # Prepare scale parameter
            scale = None
            if self.options["scale"] and self.options["scale"] != "None":
                scale = self.options["scale"]
            
            # Configure ffmpeg_config
            ffmpeg_config = {"codec": "libx264"}
            
            # Create CenterFace instance
            centerface = CenterFace(in_shape=scale)
            
            # Use video_detect from deface module directly
            class CustomVideoProcessingCallback:
                def __init__(self, thread_instance):
                    self.thread = thread_instance
                    self.frame_count = 0
                    self.total_frames = 0
                    
                def update_progress(self, progress):
                    self.thread.progress_updated.emit(progress)
                    
                def process_frame(self, frame):
                    # Increment frame counter
                    self.frame_count += 1
                    
                    # Send frame for preview
                    if hasattr(frame, 'shape'):
                        h, w = frame.shape[:2]
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        bytes_per_line = 3 * w
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        self.thread.frame_processed.emit(qt_image, self.frame_count, self.total_frames)
                    
                    # Update progress
                    if self.total_frames > 0:
                        progress = min(int((self.frame_count / self.total_frames) * 100), 99)
                        self.thread.progress_updated.emit(progress)
                    
                    # Log frame info (less frequently to avoid flooding)
                    if self.frame_count % 30 == 0:
                        if self.total_frames > 0:
                            self.thread.log_message.emit(f"Processing frame: {self.frame_count}/{self.total_frames} " +
                                                         f"({(self.frame_count/self.total_frames*100):.1f}%)")
                        else:
                            self.thread.log_message.emit(f"Processing frame: {self.frame_count}")
                    
                    return self.thread.is_running  # Return False to stop processing
            
            # Create callback instance
            callback = CustomVideoProcessingCallback(self)
            
            # Get total frames for progress tracking
            try:
                cap = cv2.VideoCapture(self.input_file)
                if cap.isOpened():
                    callback.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
            except Exception as e:
                self.log_message.emit(f"Could not determine total frames: {str(e)}")
            
            # Use our custom function to enable frame-by-frame processing with callbacks
            def custom_video_detect():
                # Initialize video reader
                import imageio
                reader = imageio.get_reader(self.input_file)
                meta = reader.get_meta_data()
                
                # Initialize video writer
                writer = imageio.get_writer(
                    self.output_file, format='FFMPEG', 
                    mode='I', fps=meta.get('fps', 30),
                    codec='libx264'
                )
                
                # Process each frame
                frame_idx = 0
                for frame in reader:
                    if not self.is_running:
                        break
                    
                    # Detect faces
                    dets, _ = centerface(frame, threshold=threshold)
                    
                    # Anonymize faces
                    deface.anonymize_frame(
                        dets, frame, mask_scale=mask_scale,
                        replacewith=replacewith, ellipse=ellipse, 
                        draw_scores=draw_scores, replaceimg=None,
                        mosaicsize=mosaicsize
                    )
                    
                    # Write frame to output
                    writer.append_data(frame)
                    
                    # Call callback
                    frame_idx += 1
                    callback.process_frame(frame)
                
                # Cleanup
                reader.close()
                writer.close()
                
                return frame_idx
            
            # Run the processing
            self.log_message.emit(f"Starting video processing with deface module...")
            try:
                frames_processed = custom_video_detect()
                self.log_message.emit(f"Video processing completed. Processed {frames_processed} frames.")
                self.progress_updated.emit(100)  # Ensure 100% at the end
                self.processing_finished.emit("Video processing completed")
            except Exception as e:
                error_msg = f"Error during video processing: {str(e)}"
                self.log_message.emit(error_msg)
                self.processing_finished.emit("Processing failed")
        
        except Exception as e:
            error_msg = f"Error during video processing: {str(e)}"
            self.log_message.emit(error_msg)
            self.processing_finished.emit(error_msg)
    
    def stop(self):
        """Stop the processing"""
        self.is_running = False


class WelcomeWindow(QMainWindow):
    """Welcome screen for the Privacy Lens application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Privacy Lens - Face Anonymization Tool")
        self.setMinimumSize(800, 600)
        
        # Set dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #222;
                color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 32px;
                text-align: center;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                color: white;
            }
        """)

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
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center everything in the layout

        # Add equal stretch at top to push content to center
        layout.addStretch(1)

        # Title
        title_label = QLabel("Privacy Lens - Face Anonymization Tool")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Subtitle
        subtitle_label = QLabel("Powered by deface library")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Description
        desc_label = QLabel("This tool allows you to anonymize faces in videos and images.\n"
                           "You can process multiple videos in batch mode and customize anonymization options.")
        desc_label.setFont(QFont("Arial", 12))
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add widgets with center alignment
        layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(40)
        layout.addWidget(desc_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(40)

        # Buttons container with center alignment
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(20)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add buttons with fixed sizes
        self.video_button = QPushButton("Video Face Anonymization")
        self.video_button.setFixedSize(400, 70)
        self.video_button.clicked.connect(self.open_video_anonymization)
        self.video_button.setStyleSheet(button_style)


        self.extract_button = QPushButton("Extract Frames from Videos")
        self.extract_button.setFixedSize(400, 70)
        self.extract_button.clicked.connect(self.open_frame_extraction)
        self.extract_button.setStyleSheet(button_style)

        self.image_button = QPushButton("Image Face Anonymization")
        self.image_button.setFixedSize(400, 70)
        self.image_button.clicked.connect(self.open_image_anonymization)
        self.image_button.setStyleSheet(button_style)

        buttons_layout.addWidget(self.video_button)
        buttons_layout.addWidget(self.extract_button)
        buttons_layout.addWidget(self.image_button)

        # Add buttons container to main layout with center alignment
        layout.addWidget(buttons_container, 0, Qt.AlignmentFlag.AlignCenter)

        # Version label
        version_label = QLabel("ACCESS Video Anonymization Tool v1.0")
        version_label.setFont(QFont("Arial", 10))
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label, 0, Qt.AlignmentFlag.AlignCenter)

        # Add equal stretch at bottom to push content to center
        layout.addStretch(1)
    
    def open_video_anonymization(self):
        try:
            self.video_window = FaceAnonymizationVideoApp()
            # Set the icon
            if hasattr(self, 'windowIcon') and not self.windowIcon().isNull():
                self.video_window.setWindowIcon(self.windowIcon())
            # Show the main screen
            self.video_window.show_main_screen()
            # Show maximized instead of fullscreen
            self.video_window.showMaximized()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open video anonymization: {str(e)}")
    
    def open_frame_extraction(self):
        """Open the frame extraction window"""
        try:
            self.extract_window = QMainWindow()
            self.extract_window.setWindowTitle("Extract Frames from Videos")
            self.extract_window.setMinimumSize(800, 600)
            
            # Create and set the central widget
            extract_widget = FrameExtractionApp()
            self.extract_window.setCentralWidget(extract_widget)
            
            # Show the window
            self.extract_window.showMaximized()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open frame extraction: {str(e)}")
    
    def open_image_anonymization(self):
        """Open the image anonymization window"""
        try:
            self.image_window = QMainWindow()
            self.image_window.setWindowTitle("Image Face Anonymization")
            self.image_window.setMinimumSize(800, 600)
            
            # Create and set the central widget
            image_widget = FaceAnonymizationBatchApp()
            self.image_window.setCentralWidget(image_widget)
            
            # Show the window
            self.image_window.showMaximized()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open image anonymization: {str(e)}")
    
    def set_window_icon(self, window):
        """Set icon for a child window"""
        if hasattr(self, 'windowIcon') and not self.windowIcon().isNull():
            window.setWindowIcon(self.windowIcon())


class VideoProcessingApp(QWidget):
    """Application for processing videos with face anonymization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Face Anonymization")
        
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Add a title label
        title = QLabel("Privacy Lens - Video Face Anonymization")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 16))
        layout.addWidget(title)
        
        # Add file selection buttons
        file_layout = QHBoxLayout()
        self.input_btn = QPushButton("Select Input Video")
        self.input_btn.clicked.connect(self.select_input_file)
        file_layout.addWidget(self.input_btn)
        
        self.output_btn = QPushButton("Select Output Location")
        self.output_btn.clicked.connect(self.select_output_file)
        file_layout.addWidget(self.output_btn)
        layout.addLayout(file_layout)
        
        # Add file path displays
        self.input_label = QLabel("No input file selected")
        layout.addWidget(self.input_label)
        
        self.output_label = QLabel("No output location selected")
        layout.addWidget(self.output_label)
        
        # Add options
        self.setup_options(layout)
        
        # Add processing button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)
        
        # Add progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Add log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        
        # Initialize variables
        self.input_file = ""
        self.output_file = ""
        self.processing_thread = None
    
    def setup_options(self, layout):
        # Add processing options
        options_group = QGroupBox("Anonymization Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["blur", "solid", "mosaic"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        options_layout.addLayout(method_layout)
        
        # Blur intensity
        blur_layout = QHBoxLayout()
        blur_label = QLabel("Blur Intensity:")
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 10)
        self.blur_slider.setValue(5)
        self.blur_slider.setTickInterval(1)
        self.blur_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        blur_layout.addWidget(blur_label)
        blur_layout.addWidget(self.blur_slider)
        options_layout.addLayout(blur_layout)
        
        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("Detection Threshold:")
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(1, 10)
        self.thresh_slider.setValue(5)
        self.thresh_slider.setTickInterval(1)
        self.thresh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        options_layout.addLayout(thresh_layout)
        
        layout.addWidget(options_group)
    
    def select_input_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file:
            self.input_file = file
            self.input_label.setText(file)
            self.log_message(f"Input file selected: {file}")
    
    def select_output_file(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "Select Output Location", "", "Video Files (*.mp4);;All Files (*)"
        )
        if file:
            if not file.endswith('.mp4'):
                file += '.mp4'
            self.output_file = file
            self.output_label.setText(file)
            self.log_message(f"Output location selected: {file}")
    
    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{timestamp}] {message}")
    
    def start_processing(self):
        if not self.input_file:
            QMessageBox.warning(self, "No Input", "Please select an input video file.")
            return
        
        if not self.output_file:
            QMessageBox.warning(self, "No Output", "Please select an output location.")
            return
        
        # Disable buttons during processing
        self.process_btn.setEnabled(False)
        self.input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        
        # Get processing options
        options = {
            "threshold": self.thresh_slider.value() / 10,  # Convert 1-10 to 0.1-1.0
            "mask_scale": 1.3,
            "anonymization_method": self.method_combo.currentText(),
            "box_method": False,  # Could add option for this
            "draw_scores": False,  # Could add option for this
            "blur_intensity": self.blur_slider.value(),
            "scale": None  # Could add option for this
        }
        
        # Start processing thread
        self.processing_thread = VideoProcessingThread(self.input_file, self.output_file, options)
        self.processing_thread.progress_updated.connect(self.progress.setValue)
        self.processing_thread.log_message.connect(self.log_message)
        self.processing_thread.processing_finished.connect(self.processing_finished)
        
        self.log_message("Starting video processing...")
        self.processing_thread.start()
    
    def processing_finished(self, message):
        self.log_message(f"Processing finished: {message}")
        self.process_btn.setEnabled(True)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application icon (will be used as default for all windows)
    app_icon = QIcon("assets/app_icon.png")  
    app.setWindowIcon(app_icon)
    
    window = WelcomeWindow()
    window.showMaximized()  # Use this instead of showFullScreen()
    sys.exit(app.exec())