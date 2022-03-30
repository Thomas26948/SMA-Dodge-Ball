from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tree():
    """
    Plot the decision tree to understand the importance of each feature
    """
        
    columns = ['mean_caught_team_1',
        'mean_caught_team_2', 'mean_precision_team_1', 'mean_precision_team_2',
        'mean_speed_team_1', 'mean_speed_team_2', 'mean_strength_team_1',
        'mean_strength_team_2']

    df = pd.read_csv('data.csv')
    print(df)
    print((df['winner']==1).sum())
    print(df.columns)
    df = df.drop(df[df['winner']==3].index)

    y = df['winner']
    X = df[columns]

    fig, ax = plt.subplots(figsize=(10,10))
    clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
    clf = clf.fit(X, y)
    tree.plot_tree(clf, feature_names=columns, filled=True, class_names=['winner','loser'], ax=ax, fontsize=9)
    plt.show()


def heatmap(real_param_1, real_param_2):
    """
    Plot the heatmap of the correlation between parameters1 and parameters2

    Args:
        real_param_1 (str): name of the first parameter
        real_param_2 (str): name of the second parameter
    """
    df = pd.read_csv('data.csv')
    

    df['remaining_players'] = df['nb_of_players_team_1'] - df['nb_of_players_team_2']
    df.rename(columns={"param1": real_param_1, "param2": real_param_2}, inplace=True)

    new_df = df[[real_param_1, real_param_2 ,'remaining_players']]
    new_df = new_df.pivot(real_param_1, real_param_2, "remaining_players")
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title("Remaining players according to the speed in both team")
    ax.set_xlabel(real_param_1)
    ax.set_ylabel(real_param_2)


    print(new_df)
    plt.imshow(new_df.values, interpolation='bicubic' )
    plt.colorbar()
    plt.show()

real_param_1 = 'strength_team_1'
real_param_2 = 'speed_team_2'
# heatmap(real_param_1, real_param_2)

plot_tree()
