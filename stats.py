from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tree():
        
    columns = ['mean_caught_team_1',
        'mean_caught_team_2', 'mean_precision_team_1', 'mean_precision_team_2',
        'mean_speed_team_1', 'mean_speed_team_2', 'mean_strength_team_1',
        'mean_strength_team_2']

    df = pd.read_csv('data.csv')
    print(df)
    print((df['winner']==1).sum())

    df = df.drop(df[df['winner']==3].index)

    y = df['winner']
    X = df[columns]

    fig, ax = plt.subplots(figsize=(10,10))
    clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
    clf = clf.fit(X, y)
    tree.plot_tree(clf, feature_names=columns, filled=True, class_names=['winner','loser'], ax=ax, fontsize=9)
    plt.show()

plot_tree()