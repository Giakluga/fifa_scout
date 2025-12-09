# ⚽ FIFA Scout - Spark & Streamlit Dashboard

![FIFA Banner](https://img.shields.io/badge/FC24-Data_Analysis-black) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![PySpark](https://img.shields.io/badge/PySpark-Powered-orange)

**FIFA Scout** is a high-performance interactive dashboard for scouting and analyzing football players, teams, and coaches.
Built using **PySpark** for big data processing and **Streamlit** for the frontend, this application is optimized to handle large datasets with minimal latency.

---

## 📊 Data Source & Engineering

The original dataset is based on the **[EA Sports FC 24 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)** by **Stefano Leone**.

To ensure optimal performance with Spark, a custom **ETL (Extract, Transform, Load) pipeline** was developed using Python:
* **Extraction:** Sourced raw CSV files (`male_players.csv`, `male_teams.csv`, `male_coaches.csv`).
* **Transformation:** Data cleaning, type casting, and schema optimization.
* **Loading:** Converted all data into **Parquet format** to leverage columnar storage and compression, significantly reducing load times compared to standard CSVs.

---

## 🚀 Key Features

### 1. 🔍 Advanced Scouting Engine
A powerful search engine allowing granular filtering of the entire database:
* **Technical Attributes:** Skills, Weak Foot, Work Rate, Positions.
* **Performance Stats:** Pace, Shooting, Passing, Dribbling, Defending, Physical.
* **Financial Data:** Market Value, Wage, Contract Expiration.
* **Team Context:** Club Rating, League, Nationality.

### 2. ⚖️ Player Comparison Tool
A dedicated tool to select up to 3 players and compare them side-by-side:
* **Radar Chart (Spider Plot):** Instant visual comparison of key stats using *Plotly*.
* **Technical Sheet:** Transposed data table for detailed analysis.

### 3. ⚡ Optimized "No-Join" Architecture
To achieve instant search results without expensive Spark JOIN operations:
* **Lazy Loading:** Heavy data processing is triggered only upon user request.
* **Python Lookup Tables:** Instead of expensive JOIN operations in Spark, the app utilizes in-memory Python Hash Maps (Dictionaries) to enrich player data with Club, League, and Coach information in real-time.
* **Smart Caching:** Extensive use of `@st.cache_resource` and `@st.cache_data`.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Backend Processing:** [PySpark](https://spark.apache.org/docs/latest/api/python/)
* **Data Engineering:** Python (Custom CSV to Parquet Script)
* **Visualization:** Plotly Graph Objects
* **Data Format:** Parquet (Columnar Storage)

---

## 📂 Project Structure

```text
fifa-scout/
├── app.py               # Main application entry point
├── utils.py             # Spark Session config & Data Loading functions
├── requirements.txt     # Python dependencies
└── data/
    └── processed/       # Optimized Parquet Datasets
        ├── fifa_player.parquet
        ├── teams.parquet
        └── coaches.parquet
     └── raw/       # CSV dataset
        ├── male_coaches.csv
        ├── male_teams.csv
        └── male_players.csv  