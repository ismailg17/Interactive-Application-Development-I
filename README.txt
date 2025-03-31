Machine Learning Model Trainer App


This is an interactive Streamlit application that allows users to train machine learning models on real-world datasets with just a few clicks


 What Does It Do?


This app is designed to help users explore, train, and evaluate machine learning models through a friendly user interface. It supports both regression and classification problems.
You can:
* Choose from built-in Seaborn datasets or upload your own CSV file
* Select features and the target variable
* Choose between two models:
   * Linear Regression (for predicting continuous values)
   * Random Forest Classifier (for classifying categories)
* Adjust key parameters like:
   * Test size
   * Number of trees (for Random Forest)
   * Train the model by clicking a single “Fit” button
* Get a full report with:
   * Regression scores (R², MSE)
   * Classification metrics (confusion matrix, classification report, ROC curve)
   * Feature importance visualization for both types of models
* Download your trained model as a `.pkl` file for future use




 How to Run It
1. Install dependencies  
   1. Create a virtual environment (optional), then run:
```bash  pip install -r requirements.txt


   2. Start the app
streamlit run app.py


      3. Open your browser — Streamlit will launch the app automatically.