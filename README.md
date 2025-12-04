# HiddenMarkovModelStreamlit  
A pipeline for training a Hidden Markov Model (HMM) on sleep and physiological data, validating it with Oura Ring exports, and visualizing nightly predictions in a Streamlit web app.

---

## 📁 Project Structure

```
Project/
├─ requirements.txt          # Python dependencies
├─ pipeline.py               # Training pipeline for the HMM
├─ validate_oura.py          # Validation pipeline for Oura Ring exports
├─ app_streamlit.py          # Streamlit UI for exploring results
├─ evaluate.py               # Shared evaluation utilities
│
├─ data_raw/                 # Raw datasets (input only)
│  ├─ dreamt_64hz/           # Training data (DREAMT dataset)
│  │  ├─ S002_whole_df.csv
│  │  ├─ S003_whole_df.csv
│  │  ├─ S004_whole_df.csv
│  │  └─ ... (8–12 subjects recommended)
│  │
│  └─ oura/                  # Validation data (Oura Ring export)
│     ├─ sleep.csv
│     ├─ readiness.csv
│     ├─ temperature.csv
│     ├─ heart_rate.csv
│     └─ ...
│
├─ data_feat/                # Auto-generated intermediate feature files
├─ models/                   # Trained model artifacts (model.pkl, labels.json)
└─ reports/                  # Validation outputs (e.g., oura_states.csv)
```

---

## 🚀 Usage Overview

### **1. Train or update the model**

```bash
python pipeline.py
```

This will:

- Load training data from `data_raw/dreamt_64hz/`
- Extract features and build parquet files in `data_feat/`
- Train a Hidden Markov Model
- Save:
  - `models/model.pkl` — the trained HMM  
  - `models/labels.json` — mapping of model states to semantic labels

---

### **2. Validate the model using Oura Ring data**

```bash
python validate_oura.py
```

This script:

- Reads Oura exports from `data_raw/oura/`
- Builds nightly features (HR, HRV, temp deviation, sleep efficiency, etc.)
- Runs the model to classify each night
- Produces a validation report such as:

```
reports/oura_states.csv
```

This file shows the model’s interpretation of each night's physiological signals.

---

### **3. Run evaluation utilities**

```bash
python evaluate.py
```

Provides diagnostics such as:

- State occupancy  
- Transition analysis  
- Fit quality  
- Debugging visualizations

---

### **4. Launch the Streamlit App**

```bash
streamlit run app_streamlit.py
```

The web application allows you to:

- Explore nightly predictions  
- Visualize rHR, RMSSD, temperature deviation, and efficiency  
- Inspect HMM state sequences  
- Interactively compare nights or subjects

---

## 📌 Notes & Tips

- `data_raw/dreamt_64hz/` is your training dataset — adding more subjects improves model generalization.
- `data_raw/oura/` is for real-world validation; drop in your own Oura exports.
- `validate_oura.py` performs robust preprocessing, including:
  - Contributors parsing  
  - rHR reconstruction from `bpm`  
  - RMSSD fallback logic  
  - Temperature deviation normalization  
- The Streamlit app automatically loads the latest model and validation results.

---
