Auteurs :  
Thibaut VAILLANT   
Cyndy Marthe Noubissi Kouopestchop  
Thomas NIGOGHOSSIAN  


## Installation librairies

Pour installer les librairies nécessaires à l'éxecution du code, veuillez utiliser la commande suivante :  
pip install -r requirements.txt


## Description du code

Le fichier dodge_ball implémente l'environnement de Dodgeball à travers une interface mesa. Pour l'executer, lancer :  
python dodge_ball.py  

Il permet de lancer un batch de 50 simulations. Chaque simulation initialise 2 équipes de manière aléatoire. Dans l'interface mesa, il est possible de modifier le nombre de 
Le script dodge_ball.py permet de sauvegarder les paramètres de l'environnement dans un fichier data.csv

Le fichier stats.py permet de visualiser un arbre de décision, et une heatmap pour comprendre l'importance des différentes caractéristiques.
python stats.py