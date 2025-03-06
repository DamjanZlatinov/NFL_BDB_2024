import pandas as pd
import numpy as np
import os

player_movements = ['gameId', 'playId', 'nflId', 'frameId', 'x', 'y', 's', 'a', 'dis', 'o', 'dir']

#change all plays to same direction
def reverse_deg(deg):
    if deg < 180:
        return deg + 180
    if deg >= 180:
        return deg - 180

def change_direction(df):
    "Direction of play should be standardized, as one play can be from left to right and another from right to left"
    df["o"]=np.where(df["playDirection"] == "left", df["o"].apply(reverse_deg), df["o"])
    df["dir"] = np.where(df["playDirection"] == "left", df["dir"].apply(reverse_deg), df["dir"])       
    df["x"] = np.where(df["playDirection"] == "left", df["x"].apply(lambda x: 120 - x), df["x"])       
    df["y"] =np.where(df["playDirection"] == "left",  df["y"].apply(lambda y: 160/3 - y), df["y"])
    return df

def who_tackled_whom(df, df_tracking, df_players):

    # Include player info for the ball carrier and we only want his movements during the play
    
    # We know who the tackler is from tracking_tackles dataset, because we used inner join on tracking with tackles.
    df_players_tackler=df_players.copy().rename(columns={'position':'position_tackler'})
    df=pd.merge(df, df_players_tackler, on='nflId')

    # We know who the ballcarrier is from plays dataset. There we have ballCarrierId
    # For the ballcarrier, we want to know his movement. We can get that from the tracking dataset.
    df_players_ballcarrier = df_players.copy().rename(columns={'nflId':'ballCarrierId', 'position':'position_ballcarrier'})
    df_players_ballcarrier_tracking = pd.merge(df_tracking[player_movements], df_players_ballcarrier, left_on=['nflId'], right_on=['ballCarrierId'])

    df=pd.merge(df, df_players_ballcarrier_tracking, on=['gameId', 'playId', 'ballCarrierId', 'frameId'], suffixes=('_tackler', '_ballcarrier'))

    return df

def load_and_merge_data(df_games, df_plays, df_players, df_tackles, df_tracking, tracking_file):

    print(f'Loading tracking data : {tracking_file}')
    
    df_tracking = change_direction(df_tracking)

    # Future work: include palyer info such as height, weight or age. For now we will only include position.
    df_players = df_players[['nflId', 'position']]
    
    # Merge datasets based on common keys
    df_tracking_plays = pd.merge(df_tracking, df_plays, on=['gameId', 'playId'], how='inner')
    df_tracking_plays_tackles =pd.merge(df_tracking_plays, df_tackles, on=['gameId', 'playId', 'nflId'], how='inner')

    df_tracking_plays_tackles['event']=np.where(df_tracking_plays_tackles['tackle']==1, 'tackle', np.where(df_tracking_plays_tackles['pff_missedTackle']==1, 'missed_tackle', np.where(df_tracking_plays_tackles['forcedFumble']==1, 'forced_fumble', (np.where(df_tracking_plays_tackles['assist']==1, 'assist', 'Other')))))

    df_merged = who_tackled_whom(df_tracking_plays_tackles, df_tracking, df_players)
    print(df_merged.shape)
    data_folder = "../data/preprocessed_data/merged_data"
    output_path = os.path.join(data_folder, f'merged_tracking_data_{tracking_file}.csv')
    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

if __name__ == "__main__":
    print('Hello')