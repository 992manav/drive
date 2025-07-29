# ML-Based Vehicle Predictive Maintenance System with Real-Time Visualization

This project implements a machine learning-based system for predicting vehicle engine conditions and provides a real-time visualization interface using Streamlit. The system uses a Gradient Boosting Classifier model to predict whether an engine is in a normal condition or requires maintenance, based on various sensor readings.

## Features

- **Predictive Maintenance:** Utilizes a machine learning model to predict engine health, enabling proactive maintenance.
- **Real-Time Visualization:** A user-friendly web interface built with Streamlit allows for real-time monitoring and prediction.
- **Customizable Input:** Users can adjust various engine parameters through sliders to see the predicted outcome.
- **Feature Descriptions:** The application provides descriptions for each engine parameter to aid understanding.

## How It Works

The system consists of two main components:

1.  **Machine Learning Model:** A `GradientBoostingClassifier` is trained on the `engine_data.csv` dataset. The data is preprocessed, and new features are engineered to improve model performance. The trained model is saved as `hhmodel.pkl`.

2.  **Streamlit Web Application:** The `app.py` file contains the Streamlit application. It loads the pre-trained model and provides a user interface with sliders for inputting engine parameters. The application then uses the model to predict the engine's condition and displays the result.

## Files in the Project

- `app.py`: The main Streamlit application file.
- `data_preprocessing.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- `engine_data.csv`: The dataset used for training the model.
- `hhmodel.pkl`: The pre-trained Gradient Boosting Classifier model.
- `README.md`: This file.

## Getting Started

### Prerequisites

- Python 3.x
- The required Python libraries, which can be installed using pip:
  ```bash
  pip install streamlit numpy pandas scikit-learn
  ```

### Running the Application

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/992manav/drive.git
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd ML-Based-Vehicle-Predictive-Maintenance-System-with-Real-Time-Visualization-main
    ```

3.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

4.  Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

- Use the sliders in the web interface to set the values for the different engine parameters.
- Click the "Predict Engine Condition" button to see the prediction.
- The application will display whether the engine condition is "Normal" or "Warning! Please investigate further."
- The "Reset Values" button will reset the sliders to their default values.

## Model Training

The `data_preprocessing.ipynb` notebook provides a detailed walkthrough of the model training process. It includes:

- Loading and exploring the `engine_data.csv` dataset.
- Feature engineering to create new, more informative features.
- Training a `GradientBoostingClassifier` model.
- Evaluating the model's performance using metrics like accuracy and a classification report.
- Saving the trained model to a `.pkl` file.
