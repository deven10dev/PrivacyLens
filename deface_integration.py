from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import cv2
import numpy as np
import imageio
import imageio.v2 as iio
import tqdm
from concurrent.futures import ThreadPoolExecutor
import functools

# Import from our package
from deface_pkg.centerface import CenterFace
from deface_pkg.utils import scale_bb, draw_det

class DefaceIntegration:
    def __init__(self):
        self.centerface = None
        self.init_detector()
    
    def init_detector(self, backend='auto'):
        """Initialize detector with hardware acceleration if available"""
        try:
            # Try CUDA first (GPU)
            self.centerface = CenterFace(in_shape=None, backend='onnxruntime', 
                                        override_execution_provider='CUDA')
            print("Using CUDA acceleration for face detection")
        except Exception as e:
            try:
                # Try TensorRT
                self.centerface = CenterFace(in_shape=None, backend='onnxruntime',
                                            override_execution_provider='TensorRT')
                print("Using TensorRT acceleration for face detection")
            except Exception as e:
                # Fall back to CPU
                self.centerface = CenterFace(in_shape=None, backend=backend)
                print("Using CPU for face detection")
    
    def anonymize_frame(self, dets, frame, mask_scale, replacewith, ellipse, draw_scores, replaceimg=None, mosaicsize=20):
        """Apply anonymization to all detected faces in a frame"""
        for i, det in enumerate(dets):
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
            # Clip bb coordinates to valid frame region
            y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
            x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
            draw_det(
                frame, score, i, x1, y1, x2, y2,
                replacewith=replacewith,
                ellipse=ellipse,
                draw_scores=draw_scores,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )
    
    def anonymize_frames_parallel(self, frames, dets_list, mask_scale, replacewith, 
                             ellipse, draw_scores, replaceimg, mosaicsize):
        """Process multiple frames in parallel"""
        def process_single(args):
            frame, dets = args
            self.anonymize_frame(dets, frame, mask_scale, replacewith, 
                                ellipse, draw_scores, replaceimg, mosaicsize)
            return frame
            
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            return list(executor.map(process_single, zip(frames, dets_list)))

    def process_image(self, 
                     input_path: str, 
                     output_path: str,
                     threshold: float = 0.2,
                     replacewith: str = 'blur',
                     mask_scale: float = 1.3,
                     ellipse: bool = True,
                     draw_scores: bool = False,
                     keep_metadata: bool = False,
                     replaceimg = None,
                     mosaicsize: int = 20) -> str:
        """Process a single image file"""
        try:
            # Load image
            frame = iio.imread(input_path)
            
            if keep_metadata:
                # Source image EXIF metadata retrieval via imageio V3 lib
                metadata = imageio.v3.immeta(input_path)
                exif_dict = metadata.get("exif", None)
            
            # Perform network inference, get bb dets but discard landmark predictions
            dets, _ = self.centerface(frame, threshold=threshold)
            
            # Apply anonymization to detected faces
            self.anonymize_frame(
                dets, frame, mask_scale=mask_scale,
                replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
                replaceimg=replaceimg, mosaicsize=mosaicsize
            )
            
            # Save the processed image
            if keep_metadata and exif_dict:
                imageio.imsave(output_path, frame, exif=exif_dict)
            else:
                imageio.imsave(output_path, frame)
                
            return output_path
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise
    
    def process_video(self, input_path, output_path, threshold=0.2, replacewith='blur',
                     mask_scale=1.3, ellipse=True, draw_scores=False, keep_audio=True,
                     ffmpeg_config=None, replaceimg=None, mosaicsize=20,
                     progress_callback=None, batch_size=4):
        """Process video with frame batching for improved performance"""
        try:
            # Open reader and writer as before
            reader = imageio.get_reader(input_path)
            meta = reader.get_meta_data()
            
            # Configure output writer
            _ffmpeg_config = ffmpeg_config or {"codec": "libx264"}
            _ffmpeg_config.setdefault('fps', meta['fps'])
            if keep_audio and meta.get('audio_codec'):
                _ffmpeg_config.setdefault('audio_path', input_path)
                _ffmpeg_config.setdefault('audio_codec', 'copy')
            
            writer = imageio.get_writer(output_path, format='FFMPEG', mode='I', **_ffmpeg_config)
            
            # Process frames in batches for better performance
            frames_batch = []
            
            for frame_i, frame in enumerate(reader):
                frames_batch.append(frame.copy())
                
                # Process when batch is full or on last frame
                if len(frames_batch) >= batch_size or frame_i == reader.count_frames() - 1:
                    # Process batch of frames
                    for i, batch_frame in enumerate(frames_batch):
                        dets, _ = self.centerface(batch_frame, threshold=threshold)
                        self.anonymize_frame(dets, batch_frame, mask_scale, replacewith,
                                            ellipse, draw_scores, replaceimg, mosaicsize)
                        writer.append_data(batch_frame)
                    
                    # Update progress
                    if progress_callback:
                        progress = min(100, int(100 * (frame_i + 1) / reader.count_frames()))
                        progress_callback(progress)
                    
                    # Clear batch
                    frames_batch = []
            
            # Close resources
            reader.close()
            writer.close()
            return output_path
        
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            raise
    
    def process_frame(self, frame, threshold=0.2, replacewith='blur', mask_scale=1.3,
                    ellipse=True, draw_scores=False, replaceimg=None, mosaicsize=20,
                    detection_scale=0.5):
        """Process a single frame with downscaling for faster detection"""
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Downscale for detection only
        if detection_scale < 1.0 and min(h, w) > 480:
            small_h, small_w = int(h * detection_scale), int(w * detection_scale)
            small_frame = cv2.resize(frame, (small_w, small_h))
            
            # Detect on smaller frame
            dets, _ = self.centerface(small_frame, threshold=threshold)
            
            # Scale detections back to original size
            if len(dets) > 0:
                scale_factor = 1.0 / detection_scale
                dets[:, :4] *= scale_factor
        else:
            # Use original frame if small enough
            dets, _ = self.centerface(frame, threshold=threshold)
        
        # Apply anonymization on original resolution
        self.anonymize_frame(dets, original_frame, mask_scale, replacewith, 
                            ellipse, draw_scores, replaceimg, mosaicsize)
        
        return original_frame

    def should_process_frame(self, frame, prev_frame, threshold=20):
        """Determine if a frame needs processing based on difference from previous"""
        if prev_frame is None:
            return True
            
        # Calculate frame difference
        diff = cv2.absdiff(frame, prev_frame)
        diff_sum = np.sum(diff)
        
        # Skip if difference is below threshold (static scene)
        return diff_sum > threshold

    @functools.lru_cache(maxsize=16)
    def get_cached_detection(self, frame_hash, threshold):
        """Cache detection results for similar frames"""
        # This would need a way to hash/identify similar frames
        pass

    def process_video_with_caching(self, input_path, output_path, threshold=0.2, 
                                  cache_similar_frames=True, **kwargs):
        """Process video with frame similarity caching"""
        # Implementation would detect similar frames and reuse results
        pass

    def get_optimized_ffmpeg_config(self, input_path):
        """Get optimized FFmpeg configuration based on input video"""
        # Default configuration
        config = {
            "codec": "libx264",
            "pixelformat": "yuv420p",
            "output_params": ["-preset", "ultrafast", "-tune", "zerolatency", "-crf", "23"]
        }
        
        # Add GPU acceleration if available
        if self._has_gpu_encoding():
            if platform.system() == "Windows":
                # NVIDIA GPU acceleration on Windows
                config = {
                    "codec": "h264_nvenc", 
                    "output_params": ["-preset", "p1", "-tune", "ll"]
                }
            elif platform.system() == "Linux":
                # Linux with VAAPI
                config = {
                    "codec": "h264_vaapi",
                    "output_params": ["-vaapi_device", "/dev/dri/renderD128"]
                }
                
        return config