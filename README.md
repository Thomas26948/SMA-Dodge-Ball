# Structure

## Resumé du jeu
2 équipes de n joueurs.
Une balle apparrait aléatoirement dans un des deux camps.
Le joueur le plus proche de la balle part la récupérer.
Il choisit une cible dans l'équipe adverse (en mouvement ou non).
Il tire. 
Si le joueur ennemi est touché, il disparait du terrain.
Sinon le processus se répète jusqu'à qu'il n'y ait plus de joueur dans une équipe.


Dans le dodgeball, je ne me souviens plus si on peut revivre une fois éliminé. Par contre si on attrape la balle en plein vol, on élimine le joueur qui a lancé la balle. Pour implémenter la bonne réception de la balle (imperceptible en 2D), on introduit une proba de bien receptionner la balle.


Le terrain est constitué de 2 zones rectangulaires de même surface, separé par une zone infranchissable. Il faudra donc définir pour chaque joueur son appartenance à une zone pour pas qu'il traverse la zone.

## Implémentation

Chaque joueur a des caractéristiques :   
• Puissance de tir ( Correspond à la vitesse de lancer de la balle)  
• Portée de tir (Jusqu'à où va la balle)  
• Taille (En 2D, on peut considérer que nos joueurs représentés par des points sont plus ou moins gros)
• Vitesse de déplacement sur le terrain
• Endurance (A implémenter si on est chaud. Il s'agit de prendre en compte qu'un joueur ne se déplace plus s'il a trop couru (et doit attendre un certain temps pour pouvoir se redéplacer))
• Précision (Un joueur avec une grande précision ne rate jamais une cible immobile. Sinon on rajoute un bruit)
• Proba de bonne réception de balle. Cette proba donne la proba qu'une balle qui tombe dans son voisinage (à la portée de ses bras) peut etre receptionner et donc éliminer le joueur qui l'a lancé.
• Taille du voisinage ? ( Pas sur de la necessité de cet attribut)


Pour le terrain, on peut soit considérer que les balles rebondissent sur les murs (dur à implémenter car il faut prendre en compte l'angle de tir etc.) Sinon on peut considérer que la balle s'arrête pile quand elle touche un mur.


Pour implémenter les tirs, le plus simple je pense c'est de calculer des equations de droite.



Est ce qu'on considère un jeu en 2D ou 3D ? 
En 2D : les tirs sont des lignes.
En 3D : visuellement sur l'interface les tirs restent des lignes mais dans le code il faudra implémenter une notion de hauteur.
Ainsi si la balle est située à une hauteur supérieure à la taille d'un joueur, on considère que le joueur n'est pas touché -> plus compliqué à représenter.

## Problèmes possibles

Une partie risque de ne jamais se terminer si les joueurs ont une trop mauvaise précision. D'où l'importance de l'endurance pour immobiliser les joueurs.  


## Création automatique des équipes.

Le prof souhaite que les équipes puivent s'auto équilibrer en fonction des différentes caractéristiques.

Idée : implémenter un score pour chaque équipe. Ce score est calculé en prenant en compte chacun des attributs des joueurs avec différents coefficient.
Par ex un joueur avec une precision de 100% vaut 1 point. S'il a une puissance de 100/100 il vaut 2 points. Par contre s'il a une vitesse de 50/100 son total est de 2.5points. Au final la somme des scores des deux équipes doit être équivalentes.
Pour déterminer les attributs des joueurs on peut alors définir un algo random ?


## Interface
### Sliders

• Mettre un slider pour choisir le nombre de joueur par équipe. (On pourrait même penser à faire 2 sliders si on veut avoir des équipes désequilibrés.)
• Mettre un slider sur le nombre de balle dans la partie

