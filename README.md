# Examen apprentissage Kilian GUILLAUME

Après avoir étudié la corrélation, 13 caractéristiques restent incluses.

## KNN

La normalisation des données a augmente fortement le score en KNN.

Après avoir essayé différentes tailles (2-10), il apparait que la taille de 3
est la meilleure pour l'apprentissage avec KNN.

## Tree

Contrairement à l'indication du client, la caractéristique Q, bien qu'ayant
une corrélation assez élevée, diminuait le score.

Les paramètres par défaut de SKLearn semblent être assez bons, car les
différentes modifications effectuées n'ont fait que réduire le score.

## MPL

Changer le nombre de layers dans le modèle ne change pas les résultats, et les
différents algorithmes n'augmentent pas le score.
