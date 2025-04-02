# PaveSense

PaveSense is a mobile application designed for real-time pothole detection using gyroscope and accelerometer data from smartphones. The application leverages machine learning and geographical visualization to detect, classify, and share pothole locations with an interactive map interface.

## Key Features

- **Real-time pothole detection** using gyroscope and accelerometer sensor data.
- Machine learning model built with **TensorFlow** for accurate road anomaly classification.
- Optimized for mobile devices using **TensorFlow Lite** for efficient, low-latency inference.
- Interactive pothole visualization and reporting using **Leaflet.js** and **OpenStreetMap**.

## Tech Stack

- **Backend/Machine Learning**: Python, TensorFlow, TensorFlow Lite
- **Frontend**: Leaflet.js, OpenStreetMap
- **Data Processing**: Numpy, Pandas
- **Mobile Sensor Data**: Gyroscope and Accelerometer

## Installation Guide

### Prerequisites

Ensure you have the following installed on your system:
- Python 3.9+
- pip

### Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ArunaK-netizen/Pothole-Detection.git
    cd Pothole-Detection
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```

3. **Install project dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    Customize the script to receive sensor data or run a sample script (not provided directly in this README).

## Project Structure

```
Pothole-Detection/ 
Pothole-Detection/
├── models/                    # Saved ML models (e.g., TensorFlow or TensorFlow Lite model files)
├── data/                      # Collected sensor data for training the CNN
├── notebooks/                 # Jupyter notebooks for experiments and analysis
├── src/                       # Core application code
│   ├── main.py                # Main script for the application
│   ├── preprocess.py          # Data preprocessing and feature extraction scripts
│   ├── ml_model.py            # TensorFlow CNN model training and optimization
│   ├── map_visualization.py   # Leaflet.js map integration
├── .gitignore                 # Ignored files
├── requirements.txt           # Dependencies for the project
├── README.md                  # Project readme
```


## How It Works

1. Collect gyroscope and accelerometer sensor data from your smartphone while driving over roads.
2. Preprocess the raw time-series data into features using Python scripts within `preprocess.py`.
3. Train a **Convolutional Neural Network (CNN)** model using the training dataset to classify road anomalies such as potholes.
4. Deploy the model on mobile devices using **TensorFlow Lite** for efficient real-time inference.
5. Visualize detected potholes on an interactive map through **Leaflet.js** and **OpenStreetMap**, sharing the information with other users.

## Demo

A GIF or screenshots showing:
- Real-time pothole detection on the app
- Interactive pothole map visualization

## Dependencies

The required Python dependencies are listed in `requirements.txt`. Key dependencies include:
- **TensorFlow**: For building and training the CNN model.
- **Numpy and Pandas**: For data processing and feature engineering.
- **Flask**: To build an optional server for data sharing (if needed).
- **Leaflet.js**: For map interface and visualization.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create your branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your changes"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact Information

For any questions or support, feel free to reach out:
- **Author**: [Aruna K](https://github.com/ArunaK-netizen)
- **Email**: arunak@example.com (Replace with your actual email)