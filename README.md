# Real-Time Facial Expression Recognition (FER)

## Project Description
This project implements a **real-time** Facial Expression Recognition (FER) system that classifies seven universal emotions: anger, disgust, fear, happiness, sadness, surprise, and neutrality.  
The system uses transfer learning with models like VGG16 and EfficientNet, along with custom CNN architectures. It is optimized for real-time performance using **TensorFlow** and **OpenCV**, addressing challenges such as lighting variations, pose, and background noise. The system is evaluated on two datasets: **CK+** (controlled) and **FER-2013** (real-world, spontaneous expressions).

## Features
- **Emotion Detection**: Classifies emotions like anger, sadness, happiness, and more.
- **Real-Time Performance**: Optimized for real-time inference using TensorFlow and OpenCV.
- **Benchmark Datasets**: Evaluated on CK+ and FER-2013 datasets.
- **Transfer Learning**: Utilizes pre-trained models (VGG16, EfficientNet) for feature extraction.
- **Custom CNN Architecture**: Designed for better performance in FER tasks.
  
## Installation

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- OpenCV 4.x
- Other dependencies (see requirements below)

### Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aymanerihane/Real-Time-Face-Expression-Recognition
   cd Real-Time-Face-Expression-Recognition
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/macOS
   venv\Scripts\activate      # For Windows
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
4. Download the pre-trained models (VGG16, EfficientNet) from <<[downlaod link](https://drive.google.com/drive/folders/1vowDVZAALaRUlM_0Alf22OtY400ZHbwV?usp=sharing)>> and place them in the models/ folder.

## Usage

  1. Run the Emotion Detection Script:
     ```bash
    streamlit run .\facial-expression-app.py

  2. Input: The script captures webcam feed and predicts the emotion from the user's facial expression in real time, also predicts the emotion for images and videos.

  3. Output: The predicted emotion is displayed on the screen.

  4. Customize: You can change the model .


This command runs the FER system using the VGG16 model.
## Results
### Performance Evaluation

    CK+ Dataset (Controlled Environment): Achieved 95% accuracy in emotion classification.

    FER-2013 Dataset (In-the-Wild): Achieved 82% accuracy due to varying lighting, poses, and expressions.

### Real-Time Inference

The system can process real-time emotion recognition with an average inference time of 50ms per frame on a standard CPU.
## Contributing

Feel free to fork this repository and submit pull requests for any enhancements or bug fixes. Contributions are always welcome!
## License

This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgements

    VGG16 Model

    EfficientNet Model

    FER-2013 Dataset




