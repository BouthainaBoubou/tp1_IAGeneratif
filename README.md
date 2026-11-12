# Activité pratique : bases de PyTorch

Ce notebook contient deux parties :

1. **Le cours** : les briques de PyTorch (tensors, autograd, modèles, boucle d'apprentissage), avec une régression et une classification simples.
2. **Les exercices** : quatre problèmes à résoudre avec PyTorch.

| Exercice | Sujet | Modèle | Résultat |
|---|---|---|---|
| 1 | Prédire un salaire | Régression linéaire `nn.Linear(2, 1)` | MAE = 381,64 DH |
| 2 | Reconnaître les chiffres MNIST | Réseau simple (MLP) | Accuracy = 97,41 % |
| 3 | Classer les fleurs Iris | MLP 4 → 16 → 3 | Accuracy = 100 % |
| 4 | Reconnaître les chiffres MNIST | Réseau convolutif (CNN) | Accuracy = 98,85 % (99,05 % avec Dropout) |

---

## Fichiers

```
Activité pratique Bases de Pytorch.ipynb   # le notebook
graphe_operations.png                      # graphe de calcul généré par torchviz
README.md
```

Ne sont **pas inclus** dans le dépôt, car ils sont recréés en exécutant le notebook :
- `data/` : le dataset MNIST, téléchargé automatiquement ;
- `*.pt` et `*.pth` : les modèles sauvegardés.

---

## Installation

```bash
pip install torch torchvision matplotlib scikit-learn torchviz torcheval torchmetrics
```

`torchviz` a aussi besoin du logiciel **Graphviz** (https://graphviz.org/download/). Le notebook ajoute `C:\Program Files\Graphviz\bin` au `PATH` : adapte ce chemin si Graphviz est installé ailleurs.

Le notebook fonctionne sur CPU. Si un GPU CUDA est disponible, PyTorch l'utilise automatiquement.

---

## Exécution

Ouvrir le notebook et exécuter les cellules **dans l'ordre**. Au premier lancement, MNIST est téléchargé dans `data/`.

---

## Partie cours

**Cycle d'apprentissage :** Données → Tensors → Modèle → Prédictions → Loss → Backward → Optimizer → Mise à jour

```python
y_pred = model(X)
loss = criterion(y_pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

| Notion | Rôle |
|---|---|
| `Tensor` | Tableau de nombres : `shape`, `dtype`, `reshape`, produit `@` |
| `device` | Calculs sur CPU ou GPU |
| `Autograd` | Calcul automatique des dérivées avec `backward()` |
| `nn.Module` / `nn.Sequential` | Construire un réseau |
| Fonctions de coût | `MSELoss` (régression), `CrossEntropyLoss` (classification) |
| Optimiseur | `Adam` ou `SGD` mettent à jour les paramètres |
| `train()` / `eval()` | Activent ou désactivent `Dropout` et `BatchNorm` |
| `Dataset` / `DataLoader` | Découper les données en lots |
| `state_dict()` | Sauvegarder et recharger un modèle |

**Résultats du cours :**
- Autograd : dL/dw = −12 et dL/db = −6, identiques au calcul à la main.
- Régression `y = 3x + 2` : le modèle trouve w = 2,989 et b = 2,030, avec R² = 0,972.
- Classification de 2 groupes de points : accuracy de 100 %, et le modèle rechargé donne les mêmes prédictions.

---

## Exercice 1 : prédire un salaire

**Objectif :** prédire le salaire mensuel à partir de l'ancienneté (0 à 20 ans) et du niveau d'expertise (1 à 5).

**Données :** 300 salariés fictifs, générés avec `Salaire = 4000 + 600 × Ancienneté + 1500 × Expertise + bruit`. Le salaire moyen est de 15 010,55 DH.

**Méthode :** 80 % pour l'entraînement et 20 % pour le test, entrées standardisées, salaires en milliers de DH, `nn.Linear(2, 1)`, SGD (lr = 0.05), 500 époques.

**Résultats :**

| | Valeur |
|---|---|
| MAE sur le test | 381,64 DH |
| Salaire prédit (7 ans, expertise 4/5) | 14 201,94 DH (attendu : 14 200 DH) |

| Coefficient | Appris | Attendu |
|---|---|---|
| Ancienneté | 592,7 DH par an | 600 |
| Expertise | 1 490,7 DH par niveau | 1 500 |
| Constante | 4 090,3 DH | 4 000 |

Le modèle retrouve bien la règle de génération. La perte ne descend pas à zéro à cause du bruit.

---

## Exercice 2 : MNIST avec un réseau simple

**Objectif :** reconnaître les chiffres manuscrits (0 à 9) avec des mini-lots.

**Modèle :** `Flatten`, `Linear(784, 128)`, `ReLU`, `Linear(128, 10)`, Adam (lr = 0.001), 5 époques.

**Résultats sur le test (10 000 images) :**

| Métrique | Valeur |
|---|---|
| Accuracy | 0,9741 |
| Recall (macro) | 0,9737 |
| Precision (macro) | 0,9744 |
| F1-score (macro) | 0,9739 |

- 259 images mal classées, dont quelques-unes sont affichées dans le notebook.
- Le modèle est sauvegardé (`mnist_mlp.pth`) puis rechargé : les prédictions sont identiques.

---

## Exercice 3 : classification des fleurs Iris

**Objectif :** prédire l'espèce d'une fleur (setosa, versicolor, virginica) à partir de 4 mesures.

**Méthode :** 120 fleurs pour l'entraînement et 30 pour le test, `StandardScaler` ajusté sur l'entraînement seulement, MLP 4 → 16 → 3, Adam (lr = 0.01), 200 époques.

**Résultats :**

| Modèle | Accuracy (test) | Perte finale |
|---|---|---|
| Réseau avec couche cachée | 100 % | 0,0432 |
| Modèle linéaire 4 → 3 | 90 % | 0,2416 |

- La couche cachée améliore nettement le résultat.
- La fleur [5.1, 3.5, 1.4, 0.2] est prédite comme **setosa**.

---

## Exercice 4 : MNIST avec un CNN

**Objectif :** reconnaître les chiffres avec un réseau convolutif.

**Architecture :**
```
Conv2d(1 → 32, 3×3) → ReLU → MaxPool(2)     28×28 → 14×14
Conv2d(32 → 64, 3×3) → ReLU → MaxPool(2)    14×14 → 7×7
Flatten (64 × 7 × 7 = 3136) → Linear(3136, 128) → ReLU → Linear(128, 10)
```

**Entraînement :** Adam (lr = 0.001), 3 époques.

**Résultats :**

| Modèle | Accuracy (test) |
|---|---|
| CNN | 98,85 % |
| CNN avec Dropout (0.5) | 99,05 % |

- Le CNN fait mieux que le réseau simple de l'exercice 2 (97,41 %), avec seulement 3 époques au lieu de 5.
- 115 images mal classées, dont 10 sont affichées dans le notebook.
- Les paramètres sont sauvegardés dans `mnist_cnn.pth`.

---

## À retenir

- La boucle `zero_grad()`, `backward()`, `step()` est la même pour tous les modèles.
- Les statistiques de normalisation se calculent **uniquement** sur l'entraînement.
- `CrossEntropyLoss` reçoit les scores bruts (logits), sans Softmax, et des étiquettes entières (`torch.long`).
- Pour les images, un CNN est plus précis qu'un réseau entièrement connecté.