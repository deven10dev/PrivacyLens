import sys
import os
import subprocess
import time
import datetime
import re    
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QWidget,
                            QProgressBar, QComboBox, QSpinBox, QCheckBox, QGroupBox,
                            QRadioButton, QButtonGroup, QMessageBox, QPlainTextEdit,
                            QListWidget, QDoubleSpinBox, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor


def detect_video_orientation(video_path):
    """Detect if a video needs rotation based on metadata"""
    try:
        # Use ffprobe to get video rotation metadata
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", 
             "-show_entries", "stream_tags=rotate", "-of", "default=noprint_wrappers=1:nokey=1", 
             video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            rotation = int(result.stdout.strip())
            if rotation == 180:
                return cv2.ROTATE_180
            elif rotation == 90:
                return cv2.ROTATE_90_CLOCKWISE
            elif rotation == 270:
                return cv2.ROTATE_90_COUNTERCLOCKWISE
    except:
        pass
        
    return None


class VideoProcessingThread(QThread):
    """Thread for extracting frames from videos without freezing the UI"""
    progress_updated = pyqtSignal(int)
    frame_extracted = pyqtSignal(str, QImage)  # frame path, image
    processing_finished = pyqtSignal(str)
    log_message = pyqtSignal(str)
    current_file_changed = pyqtSignal(str)
    
    def __init__(self, video_files, output_dir, options):
        super().__init__()
        self.video_files = video_files
        self.output_dir = output_dir
        self.options = options
        self.is_running = True
    
    def run(self):
        try:
            # Create output folder if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            total_files = len(self.video_files)
            self.log_message.emit(f"Found {total_files} videos to process")
            
            # Process each video
            for i, video_path in enumerate(self.video_files):
                if not self.is_running:
                    self.log_message.emit("Processing stopped by user")
                    self.processing_finished.emit("Processing stopped by user")
                    return
                
                video_filename = os.path.basename(video_path)
                self.current_file_changed.emit(video_filename)
                self.log_message.emit(f"Processing video {i+1}/{total_files}: {video_filename}")
                
                # Just use current time and wait a second between videos to ensure uniqueness
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_message.emit(f"Using timestamp: {timestamp_str}")
                
                # Create destination folder
                dest_dir, is_fresh = self.try_create_folders_on_timestamp(
                    timestamp_str, 
                    self.output_dir,
                    self.options.get("prefix", "HAND")
                )
                
                if not is_fresh and not self.options.get("overwrite_existing", False):
                    self.log_message.emit(f"Folder already exists for {video_filename}. Skipping...")
                    continue
                
                # Auto-detect rotation if not manually specified
                if self.options.get("rotation") is None:
                    auto_rotation = detect_video_orientation(video_path)
                    if auto_rotation is not None:
                        self.log_message.emit(f"Auto-detected rotation for {video_filename}")
                        # Use the detected rotation
                        num_frames = self.video2img(
                            video_path, 
                            dest_dir, 
                            time_intvl=self.options.get("time_interval", 1),
                            rotate_code=auto_rotation
                        )
                    else:
                        # No rotation detected, use normal processing
                        num_frames = self.video2img(
                            video_path, 
                            dest_dir, 
                            time_intvl=self.options.get("time_interval", 1)
                        )
                else:
                    # Use manually specified rotation
                    num_frames = self.video2img(
                        video_path, 
                        dest_dir, 
                        time_intvl=self.options.get("time_interval", 1),
                        rotate_code=self.options.get("rotation")
                    )
                
                self.log_message.emit(f"Extracted {num_frames} frames from {video_filename}")
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
                
                # Add a small delay before processing the next video to ensure unique timestamp
                time.sleep(1)
            
            self.log_message.emit(f"Frame extraction completed. Processed {total_files} videos.")
            self.processing_finished.emit(f"Completed processing {total_files} videos")
            
        except Exception as e:
            error_msg = f"Error during video processing: {str(e)}"
            self.log_message.emit(error_msg)
            self.processing_finished.emit(error_msg)
    
    def extract_timestamp_from_filename(self, filename):
        """Extract timestamp from filename based on different formats"""
        timestamp_str = ""
        
        # Check if it's a dropbox-style video (already renamed)
        if filename.endswith(".mov") and re.match(r"\d{8}_\d{6}\.mov", filename):
            timestamp_str = filename.split(".")[0]
            
        # Check if it's a streaming video from the app
        elif filename.startswith("video_stream_") and filename.endswith(".mp4"):
            parts = filename.split('_')
            if len(parts) == 8:
                year = parts[2]
                month = parts[3]
                day = parts[4]
                hour = parts[5]
                minute = parts[6]
                second = parts[7][:2]

                try:
                    old_date = datetime.datetime(int(year), int(month), int(day), 
                                                int(hour), int(minute), int(second))
                    timestamp_str = old_date.strftime("%Y%m%d_%H%M%S")
                except ValueError:
                    pass
        
        return timestamp_str
    
    def try_create_folders_on_timestamp(self, timestamp, parent_dir, prefix="HAND"):
        """Creates folder structure based on timestamp"""
        # Create the outer folder name and path
        outer_folder_name = f"LUAG_{timestamp}_proj"
        outer_folder_path = os.path.join(parent_dir, outer_folder_name)

        # Create the inner folder name and path
        inner_folder_name = f"{prefix}_{timestamp}"
        inner_folder_path = os.path.join(outer_folder_path, inner_folder_name)

        # Make the directories
        if os.path.exists(inner_folder_path):
            return inner_folder_path, False
        
        os.makedirs(inner_folder_path, exist_ok=True)
        self.log_message.emit(f"Created folders at: {inner_folder_path}")
        return inner_folder_path, True
    
    def video2img(self, video_path, dest_dir, time_intvl=1, rotate_code=None):
        """Extract frames from video at specific time intervals"""
        num_img = 0
        video_cap = cv2.VideoCapture(video_path)
        
        if not video_cap.isOpened():
            self.log_message.emit(f"Error: Could not open video {video_path}")
            return num_img
            
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        self.log_message.emit(f"Video FPS: {fps:.2f}")
        self.log_message.emit(f"Time interval: {time_intvl}s")
        
        frame_intvl = fps * time_intvl
        self.log_message.emit(f"Frame interval: {frame_intvl:.2f} frames")
        
        count = 0
        success = True
        
        while success and self.is_running:
            success, frame = video_cap.read()
            
            if not success:
                break
                
            count += 1
            time_stamp = count / fps
            
            if count % int(frame_intvl) == 0:
                # Apply rotation if specified
                if rotate_code is not None:
                    frame = cv2.rotate(frame, rotate_code)
                
                # Save the frame
                output_filename = os.path.join(
                    dest_dir, 
                    f"{os.path.basename(dest_dir)}_frame_{time_stamp:05.1f}s.jpg"
                )
                cv2.imwrite(output_filename, frame)
                
                # Emit signal with extracted frame for preview
                if num_img % 10 == 0:  # Only emit every 10th frame to avoid UI overload
                    # Convert to RGB for Qt
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    self.frame_extracted.emit(output_filename, qt_image)
                
                num_img += 1
        
        video_cap.release()
        return num_img
    
    def stop(self):
        """Stop the processing"""
        self.is_running = False


class FrameExtractionApp(QMainWindow):
    """Main application window for extracting frames from videos"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Frame Extraction Tool")
        self.setMinimumSize(1000, 700)
        
        self.video_files = []
        self.output_dir = ""
        self.is_processing = False
        self.processing_thread = None
        self.current_preview_file = None
        
        self.init_ui()
    
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Video file selection
        files_group = QGroupBox("Video Selection")
        files_layout = QVBoxLayout()
        
        files_btn_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Videos")
        self.add_files_btn.clicked.connect(self.browse_video_files)
        self.clear_files_btn = QPushButton("Clear List")
        self.clear_files_btn.clicked.connect(self.clear_file_list)
        files_btn_layout.addWidget(self.add_files_btn)
        files_btn_layout.addWidget(self.clear_files_btn)
        
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.load_selected_file_preview)
        
        files_layout.addLayout(files_btn_layout)
        files_layout.addWidget(QLabel("Selected Videos:"))
        files_layout.addWidget(self.file_list)
        files_group.setLayout(files_layout)
        
        # Output folder selection
        output_group = QGroupBox("Output Location")
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output folder:")
        self.output_path_label = QLabel("No folder selected")
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label, 1)
        output_layout.addWidget(self.browse_output_btn)
        output_group.setLayout(output_layout)
        
        # Extraction options
        options_group = QGroupBox("Extraction Options")
        options_layout = QVBoxLayout()
        
        # Time interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Time Interval (seconds):"))
        self.time_interval = QDoubleSpinBox()
        self.time_interval.setMinimum(0.01)
        self.time_interval.setMaximum(60)
        self.time_interval.setValue(0.2)
        self.time_interval.setSingleStep(0.1)
        self.time_interval.setDecimals(2)
        interval_layout.addWidget(self.time_interval)
        
        # Folder prefix
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Folder Prefix:"))
        self.folder_prefix = QLineEdit("HAND")
        prefix_layout.addWidget(self.folder_prefix)
        
        # Rotation options
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation:"))
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItem("None", None)
        self.rotation_combo.addItem("90° Clockwise", cv2.ROTATE_90_CLOCKWISE)
        self.rotation_combo.addItem("90° Counter-Clockwise", cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.rotation_combo.addItem("180°", cv2.ROTATE_180)
        rotation_layout.addWidget(self.rotation_combo)
        
        # Overwrite existing option
        self.overwrite_check = QCheckBox("Overwrite existing folders")
        
        # Add all layouts to options
        options_layout.addLayout(interval_layout)
        options_layout.addLayout(prefix_layout)
        options_layout.addLayout(rotation_layout)
        options_layout.addWidget(self.overwrite_check)
        options_group.setLayout(options_layout)
        
        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.current_file_label = QLabel("No file selected")
        preview_layout.addWidget(self.current_file_label)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet("background-color: #f0f0f0;")
        self.preview_label.setText("Video preview will appear here")
        preview_layout.addWidget(self.preview_label)
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
        self.process_btn = QPushButton("Extract Frames")
        self.process_btn.clicked.connect(self.toggle_processing)
        self.process_btn.setEnabled(False)
        
        self.force_stop_btn = QPushButton("Force Stop")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.clicked.connect(self.stop_processing)
        self.force_stop_btn.setStyleSheet("background-color: #ff6666;")
        
        buttons_layout.addWidget(self.process_btn)
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
        main_layout.addWidget(files_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(options_group)
        main_layout.addWidget(preview_group)
        main_layout.addLayout(progress_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(log_group)
        
        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Log application startup
        self.append_log("Video Frame Extraction Tool started")
        self.append_log("Ready to extract frames from videos")
    
    def append_log(self, message):
        """Add a message to the log with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_msg = f"[{timestamp}] {message}"
        self.log_text.appendPlainText(formatted_msg)
        
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def browse_video_files(self):
        """Open file dialog to select multiple video files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Video Files", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_paths:
            # Add selected files to the list
            for file_path in file_paths:
                # Check if file already in list
                items = [self.file_list.item(i).text() for i in range(self.file_list.count())]
                if file_path not in items:
                    self.file_list.addItem(file_path)
                    self.video_files.append(file_path)
            
            self.append_log(f"Added {len(file_paths)} video files to the list")
            
            # Reset progress bar when adding new files after completion
            if not self.is_processing:
                self.progress_bar.setValue(0)
            
            # Select first file if none selected
            if self.file_list.currentRow() == -1 and self.file_list.count() > 0:
                self.file_list.setCurrentRow(0)
            
            self.check_files_selected()
    
    def clear_file_list(self):
        """Clear the file list"""
        self.file_list.clear()
        self.video_files = []
        self.preview_label.setText("Video preview will appear here")
        self.current_file_label.setText("No file selected")
        self.progress_bar.setValue(0)
        self.check_files_selected()
        self.append_log("Cleared video list")
    
    def browse_output_folder(self):
        """Open file dialog to select output folder location"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder_path:
            self.output_dir = folder_path
            self.output_path_label.setText(os.path.basename(folder_path) if os.path.basename(folder_path) else folder_path)
            self.append_log(f"Output folder set to: {folder_path}")
            self.check_files_selected()
    
    def load_selected_file_preview(self, current, previous):
        """Load preview of the selected video file"""
        if not current:
            return
        
        file_path = current.text()
        self.current_file_label.setText(os.path.basename(file_path))
        self.current_preview_file = file_path
        
        try:
            # Open video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.preview_label.setText("Could not open video file")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get middle frame for preview
            middle_frame = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                # Convert to RGB for Qt
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
                
                # Display video info
                self.append_log(f"Video: {os.path.basename(file_path)}")
                self.append_log(f"  Resolution: {w}x{h}")
                self.append_log(f"  FPS: {fps:.2f}")
                self.append_log(f"  Duration: {duration:.2f} seconds ({frame_count} frames)")
                
            else:
                self.preview_label.setText("Could not read video frame")
            
            cap.release()
            
        except Exception as e:
            self.preview_label.setText(f"Error loading video preview: {str(e)}")
    
    def check_files_selected(self):
        """Check if videos and output folder are selected"""
        if self.file_list.count() > 0 and self.output_dir:
            self.process_btn.setEnabled(True)
        else:
            self.process_btn.setEnabled(False)
    
    def toggle_processing(self):
        """Start or stop video processing"""
        if not self.is_processing:
            self.start_processing()
        else:
            self.stop_processing()
    
    def start_processing(self):
        """Start the frame extraction process"""
        self.is_processing = True
        self.process_btn.setText("Stop Processing")
        
        # Get the list of video files from the list widget
        self.video_files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        
        # Gather options
        options = {
            "time_interval": self.time_interval.value(),
            "prefix": self.folder_prefix.text(),
            "rotation": self.rotation_combo.currentData(),
            "overwrite_existing": self.overwrite_check.isChecked()
        }
        
        # Create and start the processing thread
        self.processing_thread = VideoProcessingThread(
            self.video_files,
            self.output_dir,
            options
        )
        
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.frame_extracted.connect(self.update_preview)
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
        """Update the preview with the extracted frame"""
        pixmap = QPixmap.fromImage(image)
        
        # Scale the pixmap to fit the preview label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.preview_label.setPixmap(scaled_pixmap)
        self.current_file_label.setText(f"Extracted frame: {os.path.basename(file_path)}")
    
    def processing_finished(self, message):
        """Handle the end of processing"""
        self.is_processing = False
        self.process_btn.setText("Extract Frames")
        self.status_label.setText(message)
        
        # Disable force stop button
        self.force_stop_btn.setEnabled(False)
        
        # Re-enable UI elements
        self.disable_ui_during_processing(False)
        
        # Reset progress when finished but don't immediately show 0%
        # Only reset to 0% if the operation was explicitly stopped
        if "stopped" in message.lower():
            self.progress_bar.setValue(0)
        
        # Show a message box if completed successfully
        if "Completed" in message:
            QMessageBox.information(
                self,
                "Processing Complete",
                f"{message}\n\nOutput folder: {self.output_dir}"
            )
    
    def disable_ui_during_processing(self, disable):
        """Enable/disable UI elements during processing"""
        self.add_files_btn.setEnabled(not disable)
        self.clear_files_btn.setEnabled(not disable)
        self.browse_output_btn.setEnabled(not disable)
        self.time_interval.setEnabled(not disable)
        self.folder_prefix.setEnabled(not disable)
        self.rotation_combo.setEnabled(not disable)
        self.overwrite_check.setEnabled(not disable)
        self.file_list.setEnabled(not disable)
    
    def closeEvent(self, event):
        """Handle window close event - stop any running processes"""
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, "Processing in Progress",
                "Processing is still running. Stop processing and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.append_log("Window closing - stopping extraction...")
                
                # First disconnect all signals to prevent callbacks to destroyed objects
                try:
                    self.processing_thread.progress_updated.disconnect()
                    self.processing_thread.frame_extracted.disconnect()
                    self.processing_thread.processing_finished.disconnect()
                    self.processing_thread.log_message.disconnect()
                    self.processing_thread.current_file_changed.disconnect()
                except Exception:
                    pass  # In case signals weren't connected
                
                # Set the stop flag and wait
                self.processing_thread.is_running = False
                self.processing_thread.wait(1000)  # Wait up to 1 second
                
                # Force termination if still running
                if self.processing_thread.isRunning():
                    self.processing_thread.terminate()
            else:
                event.ignore()
                return
        
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FrameExtractionApp()
    window.show()
    sys.exit(app.exec())