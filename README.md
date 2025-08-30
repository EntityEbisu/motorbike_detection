# Motorbike Detection with YOLOv8 & Streamlit

## Project Overview
This project is a real-time motorbike detection application built with Python. It leverages a custom-trained YOLOv8 model to accurately identify motorbikes in images and videos, providing a practical demonstration of my skills in computer vision, machine learning model deployment, and web application development. The application is designed as the foundation for a more advanced system, with future plans to detect helmet violations and capture license plates for enhanced road safety monitoring.

## Key Features
- **Real-time Detection**: Utilizes a lightweight YOLOv8 model for efficient motorbike detection in various media formats.
- **Flexible Input**: Processes images and videos, with the ability to handle user uploads seamlessly.
- **Intuitive UI**: Provides a clean and easy-to-use web interface built with Streamlit, enabling anyone to test the model without any setup.
- **Annotated Output**: Generates and provides downloadable annotated images and videos, clearly showing the detection bounding boxes.
- **Cloud Deployment**: Deployed and fully functional on the Streamlit Community Cloud, demonstrating proficiency in cloud-based application deployment.

## Technical Skills Demonstrated
- **Computer Vision**: Applied core concepts of object detection and used OpenCV for image/video manipulation and annotation.
- **Machine Learning**: Implemented and fine-tuned a state-of-the-art YOLOv8 model for a specific use case.
- **Cloud Deployment**: Successfully configured a Python application for deployment on a cloud platform (Streamlit), including dependency management and handling file paths and resource limitations.
- **API Integration**: Leveraged the ultralytics library to integrate the YOLO model into a front-end application.
- **Full-Stack Development (Frontend)**: Developed a functional and responsive web interface for user interaction using Streamlit.

## Demo
Watch a demonstration of the motorbike detection application in action: [Demo Video on Google Drive](https://drive.google.com/drive/folders/13ou4Q37mBWYUZIQ8-4mXkDzWe_aYLEO0?usp=sharing).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/MotorbikeDetectionApp.git
   cd MotorbikeDetectionApp
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the App**:
   ```bash
   streamlit run source/app.py
   ```
2. **Interact with the App**: Upload an image or video, and the application will display the annotated output. You can then download the result directly from the web interface.

## Future Development & Lessons Learned
As a student project, this application has provided valuable insights and highlighted areas for future improvement.

- **Training Data Limitations**: The current model was trained on a specific dataset primarily containing front and rear views of motorbikes. As a result, its performance is limited on objects facing sideways or those that are small or far away. You can check the training notebook that I have uploaded within this repository: *motorbike_yoloV8 (2).ipynb*
- **Performance Optimization**: The inference speed can vary based on the input size and processing hardware. A key takeaway is the importance of optimizing model performance for real-world applications.

### Next Steps
- **Dataset Expansion**: Sourcing and curating a more diverse dataset to improve the model's robustness to varying angles, distances, and lighting conditions.
- **Advanced Detection**: Integrating helmet detection and license plate recognition to evolve the app from a simple detector into a more comprehensive safety monitoring tool.

## Project Details
- **Tech Stack**: Streamlit, Ultralytics YOLOv8, OpenCV, NumPy
- **Model**: `best_motorbike.pt` (trained on a custom dataset)
- **Dataset**: Custom motorbike dataset sourced from [Roboflow](https://app.roboflow.com/minh-t81tk/helmet-detection-hxqdb/models), primarily containing front and rear views of motorbikes.
- **Deployment**: Streamlit Community Cloud
- **License**: This project is licensed under the MIT License.

## Dataset
This project uses a custom motorbike dataset sourced from [Roboflow](https://app.roboflow.com/minh-t81tk/helmet-detection-hxqdb/models). The dataset primarily includes front and rear views of motorbikes, which influences the model’s performance on certain angles and distances, as noted in the "Future Development & Lessons Learned" section.

## Contact
- **Name**: Nguyen Trong Minh
- **Email**: tminh193.bil@gmail.com
