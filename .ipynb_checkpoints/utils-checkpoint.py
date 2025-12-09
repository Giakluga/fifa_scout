import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

@st.cache_resource
def get_spark_session():
    """
    Configures the Spark Session.
    """
    return SparkSession.builder \
        .appName("FIFA_Ultimate_Dashboard") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "10g") \
        .master("local[*]") \
        .getOrCreate()

def load_parquet(spark, filename):
    # 1. Get the directory where utils.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the path dynamically: ./data/processed/filename
    file_path = os.path.join(current_dir, "data", "processed", filename)
    
    # Debug info (optional, prints to console/terminal, not the app)
    # print(f"Loading: {file_path}")

    try:
        # 3. Load the dataframe
        df = spark.read.parquet(file_path)
        return df
    except Exception as e:
        # If it fails, it prints a generic error without revealing your PC username
        print(f"Error loading {filename}. Expected path: {file_path}")
        print(f"Details: {e}")
        return None

@st.cache_resource
def load_all_data():
    """
    Loads Players, Teams, and Coaches datasets using relative paths.
    """
    spark = get_spark_session()
    
    df_players = load_parquet(spark, "fifa_player.parquet")
    df_teams = load_parquet(spark, "teams.parquet")
    df_coaches = load_parquet(spark, "coaches.parquet")
    
    # OPTIMIZATION: Basic casting and immediate Caching
    if df_players:
        df_players = df_players.withColumn("value_eur", col("value_eur").cast("long")) \
                               .withColumn("wage_eur", col("wage_eur").cast("long")) \
                               .withColumn("overall", col("overall").cast("int")) \
                               .withColumn("potential", col("potential").cast("int")) \
                               .withColumn("age", col("age").cast("int"))
        
        df_players = df_players.cache()

    if df_teams:
        df_teams = df_teams.cache()
        
    if df_coaches:
        df_coaches = df_coaches.cache()

    return df_players, df_teams, df_coaches