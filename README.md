# 🌧️ Rainfall Prediction System

A machine learning project designed to predict the probability of rainfall using historical weather data. This system provides an interactive dashboard, built with Streamlit, to visualize data and make live predictions.

[**» View Live Demo**](https://rainfall-prediction-system-hq23cbe26hkk6ltymxdjiq.streamlit.app/) 

-----

## 🎯 Features

  * **Accurate Predictions:** Utilizes an XGBoost (Extreme Gradient Boosting) classifier, a powerful tree-based boosting algorithm known for its performance and accuracy.
  * **Interactive Dashboard:** A user-friendly web interface built with Streamlit to input custom weather parameters and receive instant predictions.
  * **Data Visualization:** Includes charts and graphs to visualize historical weather patterns and data distributions.
  * **Scalable:** The data preprocessing pipeline and model are built to be easily retrained with new data.

-----

## 🔧 Tech Stack

This project is built with the following technologies:

  * **Python:** The core programming language.
  * **Pandas:** For data manipulation and analysis.
  * **Scikit-learn:** For machine learning modeling and data preprocessing.
  * **Streamlit:** To create and serve the interactive web dashboard.
  * **Joblib / Pickle:** For saving and loading the trained machine learning model (`.pkl` file).

-----

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

You need to have **Python 3.8** (or newer) and **pip** installed on your system.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## 🏃‍♂️ How to Run

Once you have installed the dependencies, you can run the Streamlit app locally:

```bash
streamlit run app.py
```

Open your web browser and go to **`http://localhost:8501`** to see the application in action\!

-----

## 📂 Project Structure

Here is an overview of the key files in this project:

```
.
├── 📜 app.py                     # The main Python script for the Streamlit app
├── 🧠 rainfall_prediction_model.pkl   # The pre-trained machine learning model
├── 📄 requirements.txt          # A list of all Python dependencies
├── 📊 data/                     # (Optional) Folder for your CSV data
└── README.md                    # You are here!
```

-----

## 📈 The Model

The prediction model is an XGBoost Classifier. It's an advanced implementation of a gradient boosting algorithm, which builds a strong predictive model by combining many "weak" decision trees sequentially.

Algorithm: XGBoost (Extreme Gradient Boosting)

Dataset: Trained on the [Name of Dataset, e.g., "Rainfall data (India)"] dataset.

Features: Uses [7] features, including 'pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'.

Performance: Achieved an accuracy of [81.2]% on the held-out test set.
-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improving the model or the app, please feel free to fork the repo and create a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
