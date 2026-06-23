# Weather Forecast AI

Weather Forecast AI is a Streamlit dashboard that predicts weather conditions from temperature values and visualizes trends from a training dataset. The app uses a trained scikit-learn model saved with Joblib.

## Features

- Predicts weather condition labels such as Cold, Normal, Warm, and Hot.
- Interactive Streamlit dashboard with prediction and analytics tabs.
- Light and dark mode styling through custom CSS files.
- Dataset summary metrics and temperature charts.
- Model training script for rebuilding the classifier from CSV data.

## Tech Stack

- Python
- Streamlit
- Pandas
- scikit-learn
- Joblib
- Matplotlib

## Project Structure

```text
weather_forecast_ai/
+-- app.py                    # Streamlit web app
+-- train.py                  # Model training script
+-- forecast.py               # Experimental forecast/graph script
+-- requirements.txt          # Python dependencies
+-- styles.css                # Light theme styles
+-- styles_dark.css           # Dark theme styles
+-- data/
|   +-- weather_data.csv      # Training dataset
+-- model/
    +-- weather_model.pkl     # Trained weather model
    +-- label_encoder.pkl     # Weather label encoder
```

## Dataset Format

The training file is expected at:

```text
data/weather_data.csv
```

Required columns for the main app and training script:

```text
Timestamp, Temperature_C, Min_Temp_C, Max_Temp_C, Avg_Temp_C, Weather_Label
```

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/weather_forecast_ai.git
cd weather_forecast_ai
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Train the Model

If you update `data/weather_data.csv`, retrain the model with:

```bash
python train.py
```

This recreates:

```text
model/weather_model.pkl
model/label_encoder.pkl
```

## Push to GitHub

If the repository is not connected to GitHub yet:

```bash
git add .
git commit -m "Add weather forecast AI project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/weather_forecast_ai.git
git push -u origin main
```

If the remote already exists, use:

```bash
git add .
git commit -m "Update project README"
git push
```

## Deployment

### Recommended: Streamlit Community Cloud

This project is already a Streamlit app, so the easiest deployment target is Streamlit Community Cloud.

1. Push the project to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app from your GitHub repository.
4. Set the main file path to:

```text
app.py
```

