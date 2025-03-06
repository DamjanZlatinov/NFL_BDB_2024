import pandas as pd
import numpy as np
import os

# categorical variables 
to_one_hot = ['position_ballcarrier|mode', 'position_tackler|mode', 'offenseFormation|first']
def one_hot_encoding(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop(column, axis=1, inplace=True)
    return df

def feature_engineering(df):
    print("Generating features and preparing data for training")

    # Reduce the dimension down to tackler for each game and calculate different statistics for each play where a (not)succesful tackle was made.

    df=df.groupby(by=['gameId', 'playId', 'nflId_tackler']).agg({'offenseFormation':pd.Series.mode,'position_tackler': pd.Series.mode, 'position_ballcarrier':pd.Series.mode,'offenseFormation':'first',  'x_tackler':['min', 'max','mean', 'std', 'skew'], 'y_tackler':['min', 'max','mean', 'std', 'skew'],
                                's_tackler':['min', 'max','mean', 'std', 'skew'], 'a_tackler':['min', 'max','mean', 'std', 'skew'], 'dis_tackler':['min', 'max','mean', 'std', 'skew'],
                                'o_tackler':['min', 'max','mean', 'std', 'skew'], 'dir_tackler':['min', 'max','mean', 'std', 'skew'],
                                'quarter':['min', 'max','mean', 'std', 'skew'], 'down':['min', 'max','mean', 'std', 'skew'], 'yardsToGo':['min', 'max','mean', 'std', 'skew'],
                                'preSnapHomeScore':['min', 'max','mean', 'std', 'skew'], 'preSnapVisitorScore':['min', 'max','mean', 'std', 'skew'],
                                'passLength':['min', 'max','mean', 'std', 'skew'], 'absoluteYardlineNumber':['min', 'max','mean', 'std', 'skew'], 'defendersInTheBox': ['min', 'max','mean', 'std', 'skew'],
                                'expectedPoints':['min', 'max','mean', 'std', 'skew'], 'tackle':'max', 'assist':'max',  'pff_missedTackle':'max', 
                                'x_ballcarrier': ['min', 'max','mean', 'std', 'skew'], 'y_ballcarrier':['min', 'max','mean', 'std', 'skew'], 's_ballcarrier':['min', 'max','mean', 'std', 'skew'],
                                'a_ballcarrier':['min', 'max','mean', 'std', 'skew'], 'dis_ballcarrier': ['min', 'max','mean', 'std', 'skew'], 'o_ballcarrier':['min', 'max','mean', 'std', 'skew'], 
                                }).reset_index()
    df.columns =df.columns.map('|'.join).str.strip('|')

    df['difference_speed']=df['s_tackler|max']-df['s_ballcarrier|max']
    df['difference_a']=df['a_tackler|max']-df['a_ballcarrier|max']
    df['tackler_x_range']=df['x_tackler|max']-df['x_tackler|min']
    df['tackler_y_range']=df['y_tackler|max']-df['y_tackler|min']
    df['tackler_s_range']=df['s_tackler|max']-df['s_tackler|min']
    df['tackler_a_range']=df['a_tackler|max']-df['a_tackler|min']
    df['o_range_tackler']=df['o_tackler|max']-df['o_tackler|min']
    df['dir_range_tackler']=df['dir_tackler|max']-df['dir_tackler|min']
    df['ballcarrier_x_range']=df['x_ballcarrier|max']-df['x_ballcarrier|min']
    df['ballcarrier_y_range']=df['y_ballcarrier|max']-df['y_ballcarrier|min']
    df['ballcarrier_s_range']=df['s_ballcarrier|max']-df['s_ballcarrier|min']
    df['ballcarrier_a_range']=df['a_ballcarrier|max']-df['a_ballcarrier|min']
    df['o_range_ballcarrier']=df['o_ballcarrier|max']-df['o_ballcarrier|min']

        #create final target variable
    df['tackle_success']=np.where(((df['tackle|max']==1)), 1, 0)
    
    for col in to_one_hot:
        df = one_hot_encoding(df, col)

    print(len(df))
    print(df.info())
    print(df.tackle_success.sum())

    data_folder = "../data/preprocessed_data"
    output_path = os.path.join(data_folder, f'train_data.csv')
    df.to_csv(output_path, index=False)

    print(f"Saved merged data to {output_path}")
if __name__ == "__main__":
    print('Hello, for now')