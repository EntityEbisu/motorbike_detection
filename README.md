# Motorbike Detection App

This is a Streamlit-based application for detecting motorbikes in images, webcam feeds, and videos using a YOLOv8 model. This motorbikes-only detection model and app represent the first step towards a more robust application, with future development planned to include helmet violation detection and license plate capture.

## Features
- Detect motorbikes in uploaded images, webcam, or videos.
- Save annotated outputs as images (for images/webcam) or videos/frames (for videos).
- Log detection results for each session.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MotorbikeDetectionApp.git
   cd MotorbikeDetectionApp
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the app:
   ```bash
   streamlit run app.py
   ```
2. Select an input type (Image, Webcam, Video) and upload/process accordingly.
3. Outputs are saved to `output/`; logs are in `logs/`.

## Model
- Trained model: `best_motorbike.pt` (download from [Placeholder Link - Upload to Google Drive later]).
- Training notebook: [motorbike_detection_training.ipynb](motorbike_detection_training.ipynb) (contains training details).

## Training Data
- Sourced from a Roboflow dataset (link to be added if public).
- Description: Contains images of motorbikes, primarily front-facing and rear-facing, which impacts detection performance (see Weaknesses).

## Sample Output
- Annotated video example: [Placeholder Link - Upload demo video to Google Drive later].

## Logs
- Generated locally at `logs/detection_log_YYYYMMDD_HHMMSS.txt`.
- Format: `YYYY-MM-DD HH:MM:SS - Processed [source] - [num] motorbikes detected`.

## Deployment
- Deployed on Streamlit Community Cloud: [Deployed URL to be added once available].

## Weaknesses
- The model performs poorly with small objects (e.g., motorbikes far away) and objects facing sideways due to the training dataset only including front-facing and rear-facing images. This limitation highlights the need for a more diverse dataset in future iterations.

## Future Development
- Enhance the application to include helmet violation detection for improved safety monitoring.
- Add license plate capture functionality for regulatory compliance and tracking.
- Address current weaknesses by retraining with a dataset that includes varied angles and distances.

## License
- This project is licensed under the MIT License. See below for details:

  ```
  MIT License

  Copyright (c) [2025] [Your Name or Organization]

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

  - Replace `[2025] [Your Name or Organization]` with the appropriate copyright year and your name/organization.

## Notes
- Inference speed and detection accuracy may vary due to hardware limitations and the current training dataset.
- Contributions and feedback are welcome to improve the model and app.

```
- **Deployment**: Want to move to Streamlit Cloud next?

Let me know your progress or any assistance needed!
