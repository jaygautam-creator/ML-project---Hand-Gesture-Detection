# ML-project---Hand-Gesture-Detection
Like it's a project where who can't able to speak and other want to know what he/she saying camera auto detect gestures of both hands
# Hand Gesture Detection

This project focuses on detecting and interpreting specific hand gestures, including "Hello," "I Love You," "No," and "Yes." The system is designed to collect, process, and classify gestures using machine learning, providing real-time detection and feedback.

## Project Overview

The goal of this project is to enable a machine learning model to identify predefined hand gestures using image data. It uses computer vision and machine learning techniques to train a model that recognizes gestures and converts them into meaningful outputs.

### Features
- Detect gestures for "Hello," "I Love You," "No," and "Yes."
- Includes data collection and preprocessing scripts.
- Converts the trained model to TensorFlow Lite for lightweight deployment.
- Supports testing and real-time gesture recognition.

---

## Project Structure

```
Hand-Gesture-Detection/
├── data/                  # Dataset (raw and processed data)
├── images/                # Example images of gestures
├── models/                # Saved model files
├── collective_data.py     # Script to collect and merge data
├── convert_to_tflight.py  # Converts models to TensorFlow Lite
├── data_collection.py     # Script for collecting gesture data
├── test.py                # Script to test the trained model
├── README.md              # Project description and setup guide
├── requirements.txt       # List of required libraries
└── .gitignore             # Files/folders to exclude from Git
```

---

## Setup and Installation

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/Hand-Gesture-Detection.git
cd Hand-Gesture-Detection
```

### 2. Install Required Libraries

Ensure you have Python 3.x installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Scripts

- **Data Collection**
  Collect images for gestures:
  ```bash
  python data_collection.py
  ```

- **Training (if applicable)**
  Train the model using your dataset (add the training script here if you have one).

- **Model Conversion**
  Convert the trained model to TensorFlow Lite format:
  ```bash
  python convert_to_tflight.py
  ```

- **Testing**
  Test the model with live input or images:
  ```bash
  python test.py
  ```

---

## Gesture Details

### 1. Hello
- Open palm, fingers extended upward, waved back and forth.

### 2. I Love You
- Thumb, index, and pinky fingers extended; middle and ring fingers curled inward.

### 3. No
- Index finger moved side to side (as in shaking head).

### 4. Yes
- Fist moved up and down (as in nodding head).

---

## Dependencies

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- (Add any other dependencies your project uses)

Install these dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Example Images

Include example images or screenshots in the `images/` folder for reference.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Thanks to the open-source community and the resources that made this project possible.
