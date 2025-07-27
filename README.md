# NOx Conversion Efficiency Prediction Project

This project provides a machine learning solution to predict NOx (Nitrogen Oxide) Conversion Efficiency based on various engine and system parameters. It includes a data processing pipeline, a machine learning model for prediction, and a graphical user interface (GUI) for easy interaction.

## Project Structure

```
.
├── data/
│   ├── sample_data.xlsx              # Sample dataset for demonstration
│   └── (your_full_data.xlsx)         # Your full dataset goes here (e.g., nox_data.xlsx)
├── models/
│   ├── nox_model.pkl                 # Trained machine learning model (RandomForestRegressor)
│   ├── feature_scaler.pkl            # StandardScaler for feature scaling
│   ├── feature_list.json             # List of features used by the model
│   └── feature_importance.csv        # Feature importance scores from the trained model
├── src/
│   ├── __init__.py                   # Makes 'src' a Python package
│   ├── train_model.py                # Script for data preprocessing, model training, and evaluation
│   ├── predictor.py                  # Module for making predictions using the trained model
│   └── gui.py                        # Graphical User Interface for interactive predictions
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Setup and Installation

1.  **Clone the Repository (if applicable) or navigate to the project directory:**
    ```bash
    cd path/to/your/Nox-Project
    ```

2.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Data

The project expects your data to be in an Excel (`.xlsx`) or CSV (`.csv`) format.

*   **Sample Data**: A `sample_data.xlsx` file with 100 synthetic rows is provided in the `data/` directory for immediate testing and demonstration. This data is sufficient to train a model and test the GUI.
*   **Your Full Data**: For a truly effective prediction model, you will need to replace `data/sample_data.xlsx` or provide a new path to your full dataset (e.g., `nox_data.xlsx`) which contains 4 lakh (400,000) rows. Ensure your dataset has the same column names as specified in the original images, and includes the `NOX_CE` column as the target variable.

**Important Note on Columns:**
The model automatically detects and uses all numerical columns in your provided data (except `NOX_CE`) as features. The target column, `NOX_CE`, is what the model learns to predict.

## Model Explained: Random Forest Regressor (Like a Child!)

Imagine you want to predict a number, like the NOx Conversion Efficiency (a percentage indicating how well a system turns bad air into cleaner air).

**Why Random Forest?**
We chose the **Random Forest Regressor** for this task because it's a very powerful and versatile model, known for its accuracy and ability to handle complex, non-linear relationships in data. It's also quite robust to noisy data and can automatically figure out which factors (features) are most important.

**What is Random Forest? (Like a child explains it!)**
Think of the Random Forest as having a big team of expert guessers, instead of just one! Each "guesser" is actually a tiny decision-maker called a "Decision Tree."

1.  **Many Independent Guessers (Decision Trees):** Instead of relying on a single prediction, we train *many* (e.g., 100) different Decision Trees. Each tree learns to make its own guess based on the data.
2.  **Diverse Perspectives (Randomness):** To make sure our guessers are not all exactly alike (and make the same mistakes), we give them "random" starting points:
    *   Each tree is only allowed to look at a random *subset* of your data rows (not all 4 lakh rows).
    *   For each decision it makes, each tree is only allowed to consider a random *subset* of the available input columns (features).
    This "randomness" ensures each tree learns slightly differently, providing diverse "opinions."
3.  **Team Decision (Averaging):** When it's time to predict a new NOx CE value, we ask *all* our Decision Trees to make a guess. Then, we simply take the average of all their individual guesses. This "wisdom of the crowd" approach usually leads to a much more accurate and stable prediction than any single tree could make!

So, in simple terms, Random Forest is like consulting a whole committee of diverse experts and averaging their opinions to get the best possible answer.

## Core Components

### `src/train_model.py`

This script handles the entire machine learning pipeline:

*   **Data Loading**: Reads data from `data/sample_data.xlsx` by default. You can specify your own data file.
*   **Preprocessing**:
    *   Identifies features (input columns) and the target (`NOX_CE`).
    *   Handles duplicate columns (keeps the first occurrence).
    *   Removes columns that are entirely empty.
    *   Imputes missing numerical values with their column mean.

*   **Data Analysis**: Calculates and prints the correlation of each feature with the `NOX_CE` target, helping to understand how strongly each input parameter influences the NOx Conversion Efficiency.

*   **Data Splitting (How is it trained and tested?)**:
    Before training, the dataset is split into two parts:
    *   **Training Set (80% of data):** This is the data the model *sees* and *learns* from. It's like a student studying for an exam.
    *   **Testing Set (20% of data):** This data is kept completely separate and the model *never* sees it during training. It's used *only* to evaluate how well the model performs on new, unseen data, like a pop quiz after studying. This ensures we get an honest measure of the model's real-world performance. The split is done randomly, but reproducibly (using `random_state=42`) so you get the same split every time.

*   **Feature Scaling**:
    *   Uses `StandardScaler` to normalize feature values. This means adjusting all numerical input values so they are on a similar scale. Why do this? Imagine measuring distances in miles, kilometers, and inches all at once. It would be confusing! Scaling ensures all features contribute equally to the model's learning process and prevents features with larger numerical values from dominating.

*   **Model Training (How is it trained?)**:
    *   The `RandomForestRegressor` model is "trained" by being shown the input features and the corresponding `NOX_CE` values from the **training set**. The model adjusts its internal "rules" (the decisions within its many trees) to learn the relationships between the inputs and the target efficiency. It tries to minimize the errors between its predictions and the actual `NOX_CE` values in the training data.

*   **Model Evaluation (How is accuracy measured?)**:
    *   After training, the model's "accuracy" (performance) is measured using the **testing set** (the data it has never seen). The model makes predictions on the testing set's inputs, and these predictions are compared to the actual `NOX_CE` values in the testing set.
    *   **Mean Squared Error (MSE):** This is a common metric for regression tasks. It calculates the average of the squared differences between the predicted and actual `NOX_CE` values.
        *   **What it means:** A lower MSE means the model's predictions are, on average, closer to the actual values. It's like measuring how "off" the model's guesses are.
    *   **R-squared (R2) Score:** This metric represents the proportion of the variance in the dependent variable (`NOX_CE`) that is predictable from the independent variables (features).
        *   **What it means:** An R2 score closer to 1 (or 100%) indicates that the model explains a large portion of the variability in the `NOX_CE` and thus fits the data well. An R2 of 0 means the model explains none of the variability. A negative R2 means the model is performing worse than simply predicting the mean of the `NOX_CE` values.
        *   **Note**: With very small datasets (like the initial 5-row sample), R2 might be `NaN` and MSE might be misleading. Always use your full dataset for accurate and meaningful evaluation metrics.

*   **Model Saving**: Saves the trained model (`nox_model.pkl`), the scaler (`feature_scaler.pkl`), the list of features (`feature_list.json`), and feature importances (`feature_importance.csv`) to the `models/` directory. These saved files allow you to use the trained model later without retraining.

**How to Run `train_model.py`:**
From the project root directory:
```bash
python src/train_model.py
```
To use a different data file:
```bash
python src/train_model.py --data_path data/your_full_data.xlsx
```

### `src/predictor.py`

This module provides the core prediction functionality:

*   **Model Loading**: Loads the pre-trained model, scaler, and feature list from the `models/` directory.
*   **Dynamic Feature Information**: Automatically extracts feature descriptions and valid ranges from the `feature_list.json` and `sample_data.xlsx`.
*   **Input Validation**: Checks if prediction inputs are valid and within reasonable ranges based on the training data, preventing errors and nonsensical predictions.
*   **Prediction Function**: `predict_nox_efficiency(user_dict)` takes a dictionary of input parameters, prepares them, and returns the predicted NOx Conversion Efficiency, along with a confidence score and a performance category (Excellent, Good, Fair, Poor).

**How to Run `predictor.py` (CLI Test):**
From the project root directory:
```bash
python src/predictor.py
```
This will run a quick test using the first row of `sample_data.xlsx` and print the prediction.

### `src/gui.py`

This script provides an interactive Graphical User Interface:

*   **User-Friendly Input**: Presents a form where you can enter the engine and system parameters.
*   **Smart Input Fields**:
    *   It automatically identifies and **only displays the top 10 most important features** (based on `models/feature_importance.csv`) to simplify data entry.
    *   It provides the expected range for each input field for guidance.
*   **Hidden Parameters**: Features that are part of the model but not displayed in the GUI are automatically set to their default/midpoint values during prediction.
*   **Load Defaults/Clear All**: Buttons to easily load default (mid-range) values or clear all fields.
*   **Prediction Display**: Shows the predicted NOx Conversion Efficiency, performance category, and prediction confidence.
*   **Recommendations**: Provides basic interpretations and recommendations based on the prediction.

**How to Run `gui.py`:**
From the project root directory:
```bash
python src/gui.py
```

## Workflow Overview

1.  **Prepare Your Data**: Ensure your `nox_data.xlsx` (or other data file) is clean and placed in the `data/` directory.
2.  **Train the Model**: Run `python src/train_model.py` to train the model on your data. This will generate the necessary `.pkl`, `.json`, and `.csv` files in the `models/` directory.
3.  **Run the GUI**: Execute `python src/gui.py` to open the prediction interface.
4.  **Make Predictions**: Enter values for the important parameters in the GUI and click "Predict NOx CE" to get real-time predictions.

This project provides a robust framework for NOx Conversion Efficiency prediction, designed for ease of use and extensibility.
