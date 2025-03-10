## Face Anonymization Tool
A desktop application for automatically anonymizing faces in videos and images using the deface library.

### Overview
This tool provides a user-friendly interface for the deface library, allowing you to:

- Anonymize faces in videos with various methods (blur, solid boxes, mosaic)
Process multiple videos in batch mode
- Extract frames from videos
- Anonymize faces in batch image collections

### Prerequisites
- Python 3.7 or higher
- ffmpeg (optional but recommended for improved video handling)

### Installation
1. Clone this repository or download the source code:
```bash
git clone https://github.com/deven10dev/PrivacyLens.git
cd PrivacyLens
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv privacy_lens
```

3. Activate the virtual environment:
    - Windows:
    `privacy_lens\Scripts\activate`

    - macOS/Linux:
    `source privacy_lens/bin/activate`

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the application:
`python desktop_application.py`

### Available Tools
1. Video Face Anonymization:
    - Select one or multiple videos
    - Choose an output folder
    - Configure anonymization settings
    - Process videos in batch mode

2. Image Face Anonymization:
    - Process multiple images at once
    - Configure similar anonymization settings as video mode

3. Extract Frames from Videos:
    - Extract frames from video files at specified intervals

## Anonymization Options
- Methods: Blur, Solid boxes, Mosaic
- Detection Threshold: Control sensitivity of face detection
- Mask Scale: Adjust size of anonymization mask
- Downscale for Detection: Improve performance on large videos
- Box Method: Use rectangular instead of elliptical masks
- Draw Scores: Show detection confidence scores

## Troubleshooting
- Deface Not Found: Install the deface library using `pip install deface`
- Video Processing Issues: Try using a different video format like .avi or .mkv
- Corrupt MP4 Files: Re-encode with ffmpeg: `ffmpeg -i input.mp4 -c copy fixed.mp4`
- Detection Model Issues: Try reinstalling deface: `pip uninstall deface && pip install deface`

## License
This project is open-source and available under the [MIT LICENSE](LICENSE) terms.

## Credits
This application uses the deface library for face detection and anonymization.