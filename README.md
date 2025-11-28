# HiddenMarkovModelStreamlit
A pipeline to train a hidden markov model on sleep data and display the trained models finding on streamlit

## The application expect the following folder structure:
```
Project/
├─ requirements.txt
├─ pipeline.py
├─ validate_oura.py
├─ app_streamlit.py
├─ data_raw/   # datasets working on expanding this for model improvement
│  ├─ dreamt_64hz/ # training dataset
│  │  ├─ S002_whole_df.csv
│  │  ├─ S003_whole_df.csv
│  │  ├─ S004_whole_df.csv
│  │  └─ ... (8–12 subjects to start)
│  └─ oura/                            # using this for validation dataset
│     ├─ sleep.csv
│     └─ readiness.csv
├─ data_feat/           # created by script
├─ models/              # model.pkl, labels.json
```
## python pipeline.py 
 1. builds the model based on current datasets
 2. creates parquet files for model training

## python validate_oura.py
1. tells us how our model is performing

## streamlit run app_streamlit.py  
1. runs the current code for the webapplication
