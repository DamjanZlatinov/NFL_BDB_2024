import pandas as pd
import os
from scripts.preprocessing import load_and_merge_data
from scripts.feature_engineering import feature_engineering
from scripts.train_model import train_model
# Load datasets 

# Prepare the dataset 
def prepare_data():
    data_folder_tacking = "../data/raw_data/tracking"
    data_folder = "../data/raw_data"
    games_path = os.path.join(data_folder, 'games.csv')
    plays_path = os.path.join(data_folder, 'plays.csv')
    players_path = os.path.join(data_folder, 'players.csv')
    tackles_path = os.path.join(data_folder, 'tackles.csv')

    # Load the data
    # This files should be the same for all weeks, only the tracking data changes
    df_games = pd.read_csv(games_path)
    df_plays = pd.read_csv(plays_path)
    df_players = pd.read_csv(players_path)
    df_tackles = pd.read_csv(tackles_path)

    # Each tracking file is 1 week of data and more than 1M entries. By loading and preprocessing one by one, we can save memory.
    files = os.listdir(data_folder_tacking)
    for file_name in files:
        tracking_file = file_name[:-4]
        df_tracking = pd.read_csv(os.path.join(data_folder_tacking, file_name))
        load_and_merge_data(df_games, df_plays, df_players, df_tackles, df_tracking, tracking_file)

def create_features():
    prepared_data = "../data/preprocessed_data/merged_data"
    df = pd.concat([pd.read_csv(os.path.join(prepared_data, file)) for file in os.listdir(prepared_data) if file.endswith(".csv")], ignore_index=True)
    feature_engineering(df)

def train_and_save_model():
    train_data = "../data/preprocessed_data"
    df = pd.read_csv(os.path.join(train_data, 'train_data.csv'))
    train_model(df)

if __name__ == "__main__":
    # If i decide to change something in the future engineering part, i can call only that pipeline (function here).
    # prepare_data()
    # create_features()
    train_and_save_model()


   