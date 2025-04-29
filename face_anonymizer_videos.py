import platform
import sys
import os
import subprocess
from pathlib import Path
import time
import datetime
import cv2
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QWidget,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QSlider,
                            QProgressBar, QComboBox, QSpinBox, QCheckBox, QGroupBox,
                            QRadioButton, QButtonGroup, QMessageBox, QPlainTextEdit,
                            QListWidget, QListWidgetItem, QStackedWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QIcon
from centerface import CenterFace
import deface
import imageio
import cv2

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
            # Check file size before processing
            file_size_mb = os.path.getsize(self.input_file) / (1024 * 1024)
            if file_size_mb > 500:  # 500MB threshold
                message = f"Large file detected ({file_size_mb:.1f} MB). Processing may take a long time."
                self.log_message.emit(message)
                if file_size_mb > 2000:  # 2GB threshold
                    answer = QMessageBox.question(
                        None, 
                        "Very Large File", 
                        f"The file is extremely large ({file_size_mb:.1f} MB). Processing may take a very long time and use significant system resources. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if answer == QMessageBox.StandardButton.No:
                        return

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
            blur_intensity = self.options["blur_intensity"] if "blur_intensity" in self.options else 5
            
            # Prepare scale parameter
            scale = None
            if self.options["scale"] and self.options["scale"] != "None":
                scale_parts = self.options["scale"].split('x')
                if len(scale_parts) == 2:
                    try:
                        scale = (int(scale_parts[0]), int(scale_parts[1]))
                    except ValueError:
                        scale = None
            
            # Create CenterFace instance
            centerface = CenterFace(in_shape=scale)
            
            # Get total frames for progress tracking
            try:
                import cv2  # Make sure cv2 is imported here
                cap = cv2.VideoCapture(self.input_file)
                if not cap.isOpened():
                    self.log_message.emit("Warning: Could not open input video file.")
                    total_frames = 0
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        self.log_message.emit("Warning: Could not determine total frames from metadata.")
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
            
            # Log all parameters for debugging
            self.log_message.emit(f"Processing parameters:")
            self.log_message.emit(f"  Anonymization method: {replacewith}")
            self.log_message.emit(f"  Threshold: {threshold}")
            self.log_message.emit(f"  Mask scale: {mask_scale}")
            self.log_message.emit(f"  Box method: {self.options['box_method']}")
            self.log_message.emit(f"  Draw scores: {draw_scores}")
            if replacewith == "mosaic":
                self.log_message.emit(f"  Mosaic size: {mosaicsize}")
            if replacewith == "blur":
                self.log_message.emit(f"  Blur intensity: {blur_intensity}")

            # Process the video using imageio and deface directly
            try:
                # Initialize video reader
                reader = imageio.get_reader(self.input_file)
                meta = reader.get_meta_data()
                fps = meta.get('fps', 30)
                
                # Configure video writer with explicit orientation control
                try:
                    # Initialize video reader
                    reader = imageio.get_reader(self.input_file)
                    meta = reader.get_meta_data()
                    fps = meta.get('fps', 30)
                    
                    # Get first frame to determine dimensions
                    try:
                        test_frame = reader.get_data(0)
                        height, width = test_frame.shape[:2]
                        self.log_message.emit(f"Video dimensions: {width}x{height}")
                    except Exception as e:
                        self.log_message.emit(f"Could not read first frame: {str(e)}")
                        height, width = None, None
                    
                    # Close and reopen reader to start from beginning
                    reader.close()
                    reader = imageio.get_reader(self.input_file)
                    
                    # Configure ffmpeg options with explicit dimensions
                    ffmpeg_config = {
                        "codec": "libx264",
                        "macro_block_size": 1,  # Set to 1 to avoid resizing
                        "ffmpeg_log_level": "warning",
                    }
                    
                    if height is not None and width is not None:
                        ffmpeg_config["output_params"] = [
                            "-pix_fmt", "yuv420p",
                            "-vf", f"scale={width}:{height}"  # Ensure exact dimensions
                        ]
                except Exception as e:
                    self.log_message.emit(f"Error configuring video dimensions: {str(e)}")
                    ffmpeg_config = {"codec": "libx264"}
                
                # Configure ffmpeg options
                ffmpeg_config = {"codec": "libx264"}
                
                # Initialize video writer
                writer = imageio.get_writer(
                    self.output_file, 
                    format='FFMPEG', 
                    mode='I', 
                    fps=fps,
                    **ffmpeg_config
                )
                
                # Process each frame
                frame_count = 0
                last_progress_update = time.time()
                last_heartbeat = time.time()
                
                self.log_message.emit(f"Starting video processing with direct deface module integration...")
                
                for frame in reader:
                    if not self.is_running:
                        self.log_message.emit("Processing stopped by user")
                        break
                    
                    # Detect faces using centerface
                    dets, _ = centerface(frame, threshold=threshold)
                    
                    # Anonymize faces
                    if replacewith == "blur":
                        # For blur method, handle intensity directly
                        # Exponentially increased blur kernel size for much stronger effect
                        blur_kernel_size = max(5, int(501 - (blur_intensity ** 4) * 0.05))
                        # Make sure kernel size is odd
                        if blur_kernel_size % 2 == 0:
                            blur_kernel_size += 1
                        
                        self.log_message.emit(f"Using blur kernel size: {blur_kernel_size}")
                        
                        # Process each detected face
                        for det in dets:
                            x1, y1, x2, y2, _ = det
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Calculate mask scale (expand detection box)
                            width, height = x2-x1, y2-y1
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            # Apply scaling factor
                            new_width = int(width * mask_scale)
                            new_height = int(height * mask_scale)
                            
                            # Recalculate box coordinates with mask_scale
                            x1_scaled = max(0, center_x - new_width // 2)
                            y1_scaled = max(0, center_y - new_height // 2)
                            x2_scaled = min(frame.shape[1], center_x + new_width // 2)
                            y2_scaled = min(frame.shape[0], center_y + new_height // 2)
                            
                            # Extract face region (with scaling)
                            face_region = frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled].copy()
                            
                            # Skip if face region is empty (can happen at image borders)
                            if face_region.size == 0:
                                continue
                                
                            # Apply blur with appropriate kernel size
                            blurred_face = cv2.GaussianBlur(face_region, (blur_kernel_size, blur_kernel_size), 0)
                            
                            # Apply multiple passes based on intensity (more passes for lower intensity values)
                            additional_passes = max(1, 10 - blur_intensity)
                            for _ in range(additional_passes):
                                blurred_face = cv2.GaussianBlur(blurred_face, (blur_kernel_size, blur_kernel_size), 0)

                            # For intensity 1-3, add pixelation on top of blurring for maximum anonymization
                            if blur_intensity <= 3:
                                # Add pixelation effect on top of blur
                                height, width = blurred_face.shape[:2]
                                pixel_size = 12 - blur_intensity * 2  # Larger pixels for stronger effect
                                temp = cv2.resize(blurred_face, (width // pixel_size, height // pixel_size), 
                                                  interpolation=cv2.INTER_LINEAR)
                                blurred_face = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
                                
                            self.log_message.emit(f"Applied {additional_passes+1} blur passes with kernel size {blur_kernel_size}")
                            if blur_intensity <= 3:
                                self.log_message.emit(f"Added pixelation effect for maximum privacy")
                            
                            # Replace region in the frame
                            if ellipse:
                                # Create a mask for elliptical blur
                                mask_height, mask_width = y2_scaled-y1_scaled, x2_scaled-x1_scaled
                                mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
                                
                                # Draw ellipse on the mask
                                try:
                                    center = (mask_width // 2, mask_height // 2)
                                    axes = (int(mask_width // 2 * 0.95), int(mask_height // 2 * 0.95))
                                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                                    
                                    # Expand mask to match face_region channels
                                    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                                    
                                    # Apply elliptical blur
                                    frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = np.where(
                                        mask_3d > 0, 
                                        blurred_face, 
                                        frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
                                    )
                                except Exception as e:
                                    # Fallback to rectangular blur if ellipse fails
                                    self.log_message.emit(f"Ellipse error: {str(e)}, falling back to rectangle")
                                    frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = blurred_face
                            else:
                                # Apply rectangular blur
                                frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = blurred_face
                    else:
                        # For other methods, use the standard anonymize_frame
                        deface.anonymize_frame(
                            dets, frame, mask_scale=mask_scale,
                            replacewith=replacewith, ellipse=ellipse, 
                            draw_scores=draw_scores, replaceimg=None, 
                            mosaicsize=mosaicsize
                        )
                    
                    # Write the processed frame
                    writer.append_data(frame)
                    
                    # Update progress
                    frame_count += 1
                    current_time = time.time()
                    
                    # Heartbeat message every 10 seconds
                    if current_time - last_heartbeat > 10:
                        last_heartbeat = current_time
                        self.log_message.emit(f"Still processing... (current frame: {frame_count})")
                    
                    # Update progress bar roughly every second
                    if current_time - last_progress_update > 1.0:
                        last_progress_update = current_time
                        
                        # Update UI with progress
                        if total_frames > 0:
                            progress = min(int((frame_count / total_frames) * 100), 99)
                            self.progress_updated.emit(progress)
                        
                        # Log frame info (less frequently)
                        if frame_count % 100 == 0:
                            if total_frames > 0:
                                self.log_message.emit(f"Processing frame: {frame_count}/{total_frames} " +
                                                    f"({(frame_count/total_frames*100):.1f}%)")
                            else:
                                self.log_message.emit(f"Processing frame: {frame_count}")
                    
                    # Send frame for preview (every 5th frame to avoid GUI slowdown)
                    if frame_count % 5 == 0:
                        h, w = frame.shape[:2]
                        rgb_frame = frame  # imageio already gives us RGB format
                        bytes_per_line = 3 * w
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_processed.emit(qt_image, frame_count, total_frames)
                
                # Close reader and writer
                reader.close()
                writer.close()
                
                # Successful completion
                self.log_message.emit(f"Video processing completed successfully. Processed {frame_count} frames.")
                self.progress_updated.emit(100)  # Ensure 100% at the end
                self.processing_finished.emit("Video processing completed")
                
            except Exception as e:
                error_msg = f"Error during video processing: {str(e)}"
                self.log_message.emit(error_msg)
                
                # Check for common errors and provide helpful feedback
                if "No such file" in str(e):
                    self.log_message.emit("This error often occurs when the output directory doesn't exist or isn't writable.")
                    self.log_message.emit("Make sure the output path is valid and you have write permissions.")
                elif "moov atom not found" in str(e):
                    self.log_message.emit("This error often occurs with corrupted MP4 files or files with metadata issues.")
                    self.log_message.emit("Try using a different video format like .avi or .mkv")
                
                self.processing_finished.emit("Processing failed")
            
        except Exception as e:
            error_msg = f"Error during video processing setup: {str(e)}"
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
            "This tool allows you to anonymize faces in videos.\n"
            "You can process multiple videos in batch mode and customize anonymization options."
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addSpacing(30)
        
        # Get started button
        self.start_button = QPushButton("Get Started")
        self.start_button.setMinimumSize(200, 50)
        self.start_button.setFont(QFont("Arial", 14))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Version info at bottom
        layout.addStretch()
        version_label = QLabel("ACCESS Video Anonymization Tool v1.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)
        
        self.setLayout(layout)


class FaceAnonymizationVideoApp(QMainWindow):
    """Main application window for processing videos using deface library"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Face Anonymization (powered by deface)")
        self.setMinimumSize(1000, 700)
        
        self.input_file = ""
        self.output_file = ""
        self.is_processing = False
        self.processing_thread = None
        
        # Check if deface is installed
        try:
            # Import version directly from the deface module
            from version import __version__ as deface_version
            self.deface_version = deface_version
        except ImportError:
            # Fallback if version module isn't available
            self.deface_version = "unknown"
            # Don't exit - we can still use the module even if we can't get its version
            
        # Also check for ffmpeg (optional but helpful)
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
        
        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        
        # Create welcome screen
        self.welcome_screen = WelcomeScreen()
        self.welcome_screen.start_button.clicked.connect(self.show_main_screen)
        
        # Create main app widget
        self.main_app_widget = QWidget()
        
        # Add screens to stacked widget
        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.main_app_widget)
        
        # Set central widget to stacked widget
        self.setCentralWidget(self.stacked_widget)
        
        # Initialize the main UI (but stay on welcome screen)
        self.init_main_ui()
        
        # Start on welcome screen
        self.stacked_widget.setCurrentIndex(0)

    def show_main_screen(self):
        """Switch from welcome screen to main app screen"""
        self.stacked_widget.setCurrentIndex(1)
        # Log application startup after welcome screen
        self.append_log(f"Video Face Anonymization App started (powered by deface {self.deface_version})")
        self.append_log("Ready to process videos")
        
    def init_main_ui(self):
        """Initialize the main UI components"""
        main_layout = QVBoxLayout()
        
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
        
        # Add all layouts to options
        options_layout.addLayout(anon_layout)
        options_layout.addLayout(self.mosaic_layout)
        options_layout.addLayout(thresh_layout)
        options_layout.addLayout(mask_layout)
        options_layout.addLayout(scale_layout)

        # Add this code in the init_main_ui method after the checks_layout section

        # Blur intensity slider (only visible when blur method is selected)
        self.blur_intensity_layout = QHBoxLayout()
        self.blur_intensity_layout.addWidget(QLabel("Blur Intensity:"))
        self.blur_intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_intensity_slider.setMinimum(1)  # Strongest blur
        self.blur_intensity_slider.setMaximum(10) # Lightest blur
        self.blur_intensity_slider.setValue(5)    # Default: medium blur
        self.blur_intensity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.blur_intensity_slider.setTickInterval(1)  # Show all tick marks
        self.blur_intensity_value_label = QLabel("5")
        self.blur_intensity_slider.valueChanged.connect(self.update_blur_intensity_value)
        self.blur_intensity_layout.addWidget(self.blur_intensity_slider)
        self.blur_intensity_layout.addWidget(self.blur_intensity_value_label)
        blur_note = QLabel("(1=strongest blur, 10=lightest blur)")
        blur_note.setStyleSheet("color: gray; font-size: 9pt;")
        self.blur_intensity_layout.addWidget(blur_note)

        # Add the layout to options_layout after checks_layout
        options_layout.addLayout(self.blur_intensity_layout)
        options_group.setLayout(options_layout)
        
        # Update UI to hide mosaic settings initially
        self.update_ui_based_on_method("blur")
        
        # Preview
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout()

        # Create a container widget to better control sizing
        preview_container = QWidget()
        preview_container_layout = QVBoxLayout(preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)
        preview_container.setMinimumSize(640, 360)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("background-color: #f0f0f0;")
        self.preview_label.setText("Video preview will appear here during processing")

        preview_container_layout.addWidget(self.preview_label)
        preview_layout.addWidget(preview_container)
        preview_group.setLayout(preview_layout)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        # Add a batch process button
        self.batch_process_btn = QPushButton("Process All Videos")
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        self.batch_process_btn.setEnabled(False)
        buttons_layout.addWidget(self.batch_process_btn)

        self.force_stop_btn = QPushButton("Force Stop")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.clicked.connect(self.stop_processing)
        self.force_stop_btn.setStyleSheet("background-color: #ff6666;")
        
        buttons_layout.addWidget(self.force_stop_btn)
        
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
        self.main_app_widget.setLayout(main_layout)
        
        # Log application startup
        self.append_log(f"Video Face Anonymization App started (powered by deface {self.deface_version})")
        self.append_log("Ready to process videos")
    
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
            
            # Reset progress bar when adding new files
            self.progress_bar.setValue(0)
            
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
            self.progress_bar.setValue(0)
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
        # self.process_btn.setEnabled(False)
        
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
            "box_method": False,  # Always use ellipse masks
            "draw_scores": False,  # Never draw scores
            "scale": self.scale_combo.currentText() if self.scale_combo.currentIndex() > 0 else "",
            "blur_intensity": self.blur_intensity_slider.value() if self.anon_method.currentText() == "blur" else 5
        }
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
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
        
        # Reset progress bar to 0
        self.progress_bar.setValue(0)
        
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
        # self.process_btn.setText("Process Video")
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

    # Make sure to modify the original stop_processing method to handle batch mode
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

    # Update the disable_ui method to also disable batch controls
    def disable_ui_during_processing(self, disable):
        """Enable/disable UI elements during processing"""
        # self.browse_input_btn.setEnabled(not disable)
        self.browse_output_btn.setEnabled(not disable)
        self.browse_multiple_btn.setEnabled(not disable)
        self.anon_method.setEnabled(not disable)
        self.mosaic_size.setEnabled(not disable)
        self.threshold_slider.setEnabled(not disable)
        self.mask_scale_slider.setEnabled(not disable)
        self.scale_combo.setEnabled(not disable)
        self.blur_intensity_slider.setEnabled(not disable)

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
    
        # Show blur intensity only when blur method is selected
        for i in range(self.blur_intensity_layout.count()):
            item = self.blur_intensity_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(method == "blur")
    
    def update_threshold_value(self):
        """Update the threshold value label"""
        value = self.threshold_slider.value() / 100
        self.threshold_value_label.setText(f"{value:.2f}")
    
    def update_mask_scale_value(self):
        """Update the mask scale value label"""
        value = self.mask_scale_slider.value() / 10
        self.mask_scale_value_label.setText(f"{value:.1f}")

    # Add this method after update_mask_scale_value

    def update_blur_intensity_value(self):
        """Update the blur intensity value label"""
        value = self.blur_intensity_slider.value()
        self.blur_intensity_value_label.setText(f"{value}")
        self.blur_intensity_value_label.setStyleSheet("font-weight: bold;")
    
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
    
    # def check_files_selected(self):
    #     """Check if both input and output files are selected"""
    #     if self.input_file and self.output_file:
    #         self.process_btn.setEnabled(True)
    #     else:
    #         self.process_btn.setEnabled(False)
    
    def toggle_processing(self):
        """Start or stop video processing"""
        if not self.is_processing:
            self.start_processing()
        else:
            self.stop_processing()
    
        options = {
            "threshold": float(self.threshold_value_label.text()),
            "mask_scale": float(self.mask_scale_value_label.text()),
            "anonymization_method": self.anon_method.currentText(),
            "mosaic_size": self.mosaic_size.value(),
            "box_method": False,  # Always use ellipse masks
            "draw_scores": False,  # Never draw scores
            "scale": self.scale_combo.currentText() if self.scale_combo.currentIndex() > 0 else "",
            "blur_intensity": self.blur_intensity_slider.value() if self.anon_method.currentText() == "blur" else 5
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
        # Update status with simplified progress information
        if total_frames > 0 and current_frame > 0:
            self.status_label.setText(f"Processing: {(current_frame/total_frames*100):.1f}%")
        else:
            self.status_label.setText("Processing...")
        
        # Only update the image if we received a valid one
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            
            # Get the current size of the preview area with some margin
            preview_size = self.preview_label.size()
            available_width = preview_size.width() - 20  # 10px margin on each side
            available_height = preview_size.height() - 20
            
            # Calculate the scaling needed to fit the image
            img_width = pixmap.width()
            img_height = pixmap.height()
            
            # Calculate scale factors to fit width and height
            w_scale = available_width / img_width
            h_scale = available_height / img_height
            
            # Use the smaller scale to ensure the entire image fits
            scale_factor = min(w_scale, h_scale)
            
            # Scale the pixmap while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                int(img_width * scale_factor),
                int(img_height * scale_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def closeEvent(self, event):
        """Handle window close event - stop any running processes"""
        if self.is_processing and hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
            # 1. Add confirmation dialog before attempting to stop
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Video processing is still running. Do you want to stop processing and close the window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.No or reply == QMessageBox.StandardButton.Cancel:
                self.append_log("Window close canceled by user")
                event.ignore()
                return
            
            try:
                self.append_log("Window closing - stopping video processing...")
                
                # Simpler approach to disconnect signals - disconnect all slots
                try:
                    if hasattr(self.processing_thread, 'progress_updated'):
                        self.processing_thread.progress_updated.disconnect()
                    if hasattr(self.processing_thread, 'frame_processed'):
                        self.processing_thread.frame_processed.disconnect()
                    if hasattr(self.processing_thread, 'processing_finished'):
                        self.processing_thread.processing_finished.disconnect()
                    if hasattr(self.processing_thread, 'log_message'):
                        self.processing_thread.log_message.disconnect()
                except Exception as e:
                    self.append_log(f"Signal disconnect: {str(e)}")
                
                # Stop the thread safely
                self.processing_thread.is_running = False  # Make sure the flag is set
                self.processing_thread.stop()
                self.processing_thread.wait(1500)
                
                # Force termination if still running
                if self.processing_thread.isRunning():
                    self.processing_thread.terminate()
                    self.processing_thread.wait(300)
                    
            except Exception as e:
                self.append_log(f"ERROR during shutdown: {str(e)}")
        
        # Log final closure and accept event
        if hasattr(self, 'append_log'):
            self.append_log("Application closing")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = FaceAnonymizationVideoApp()
    window.show()
    sys.exit(app.exec())