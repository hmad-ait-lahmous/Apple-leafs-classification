# Classification automatique de l’état sanitaire des feuilles de pommier à l’aide du deep learning

**Projet : AppleLeaf — analyseur de santé foliaire assisté par IA**

---

## Résumé

Ce dépôt présente une application web permettant de classifier automatiquement des photographies de feuilles de pommier en trois catégories : feuille saine, symptômes de taches noires (alternariose), et symptômes de taches brunes. Le système repose sur un réseau convolutif profond construit à partir de MobileNetV2 pré-entraîné sur ImageNet, affiné par transfert d’apprentissage sur un jeu d’images locales. Une interface Flask offre le chargement d’images ou l’analyse via la webcam.

**Mots-clés :** classification d’images, transfert d’apprentissage, MobileNetV2, Flask, agriculture de précision.

---

## 1. Introduction et problématique

La détection précoce des maladies foliaires chez le pommier permet d’orienter les traitements et de limiter les pertes de rendement. L’identification visuelle reste toutefois dépendante de l’expertise et du temps disponible sur le terrain. Les approches par apprentissage profond appliquées à la vision par ordinateur offrent une voie pour automatiser une première analyse à partir de photographies grand public.

Le présent travail met en œuvre un pipeline complet : préparation des données, entraînement du modèle, puis mise à disposition des prédictions via une application locale accessible dans le navigateur.

---

## 2. Jeu de données et classes cibles

Les images sont organisées sous la forme attendue par `ImageDataGenerator` de Keras : répertoires `train/` et `test/` contenant une classe par dossier. Les trois classes considérées sont les suivantes.

- **Apple Normal** : feuille jugée saine ; les bonnes pratiques culturales peuvent être maintenues.
- **Apple Black Spot** : lésions compatibles avec une infection fongique, notamment associée à *Alternaria mali* ; un avis phytosanitaire professionnel et un éventuel traitement fongicide peuvent être nécessaires.
- **Apple Brown Spot** : taches pouvant correspondre à des lésions bactériennes ou fongiques ; des mesures culturales (aération, drainage) sont souvent recommandées en complément.

Les effectifs approximatifs du jeu d’entraînement mentionnés dans la structure du projet sont d’environ 469 images (black spot), 1060 (brown spot) et 824 (normal), réparties entre sous-dossiers dédiés. Le dossier complet `leafs/` n’est pas versionné dans ce dépôt en raison de sa taille ; il doit être restauré localement ou fourni parallèlement pour reproduire l’entraînement.

---

## 3. Méthodologie

### 3.1 Architecture

Le modèle utilise **MobileNetV2** comme extracteur de caractéristiques initial, pré-entraîné sur le corpus ImageNet. Ce choix privilégie un compromis entre précision et coût de calcul, adapté à un déploiement sur matériel modeste.

### 3.2 Stratégie d’entraînement

L’entraînement est mené en **deux phases** : d’abord apprentissage de la tête de classification uniquement (base convolutionnelle gelée, 10 époques), puis **fine-tuning** des trente couches supérieures du réseau de base (20 époques), avec des taux d’apprentissage différenciés.

### 3.3 Prétraitement et augmentation

Les entrées sont redimensionnées en **224×224** pixels, trois canaux RVB. Des transformations aléatoires (rotation, retournement, zoom, variation de luminosité et décalages) augmentent artificiellement la diversité du jeu d’entraînement et aident à limiter le surapprentissage.

### 3.4 Fichiers produits

Après exécution du script `train_model.py`, le répertoire `model/` contient notamment `apple_leaf_model.h5` (modèle sérialisé utilisé par l’application), `class_indices.json` (correspondance indices–noms de classes) et des fichiers d’historique d’entraînement au format CSV.

---

## 4. Application web

Le serveur est implémenté avec **Flask** (`app.py`). L’utilisateur peut soumettre une image ou activer la **webcam** ; le modèle TensorFlow charge le fichier `model/apple_leaf_model.h5` au démarrage. Les ressources statiques (HTML, CSS, JavaScript) gèrent l’interface et l’expérience utilisateur.

Pour des résultats fiables, il est recommandé d’utiliser une photo nette, bien éclairée, centrée sur une feuille isolée ; en mode webcam, cadrer la feuille dans la zone indiquée à l’écran réduit les erreurs de cadrage.

---

## 5. Installation et reproduction

### 5.1 Dépendances

```bash
pip install -r requirements.txt
```

### 5.2 Entraînement du modèle

Une fois le jeu de données `leafs/` en place :

```bash
python train_model.py
```

La durée dépend du matériel (ordre de grandeur : dizaines de minutes).

### 5.3 Lancement de l’application

```bash
python app.py
```

Puis ouvrir [http://localhost:5000](http://localhost:5000) dans un navigateur.

### 5.4 Arborescence utile du dépôt

```
entr/
├── leafs/                 # données (non versionnées — à ajouter localement)
├── model/                 # sorties d’entraînement (poids volumineux ignorés par Git)
├── static/
├── templates/
├── train_model.py
├── app.py
└── requirements.txt
```

---

## 6. Conclusion

Ce projet illustre une chaîne classique en vision par ordinateur appliquée à l’agronomie : transfert d’apprentissage à partir d’un backbone léger, entraînement supervisé sur des classes de symptômes, et exposition des résultats via une interface web simple. Les prédictions restent **indicatives** ; une décision de traitement sur une exploitation doit s’appuyer sur un diagnostic agronomique expert et sur la réglementation locale des produits phytosanitaires.

---

## Bibliographie

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet : A Large-Scale Hierarchical Image Database. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. https://doi.org/10.1109/CVPR.2009.5206848

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andrreetto, M., & Adam, H. (2017). MobileNets : Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv preprint* arXiv:1704.04861. https://arxiv.org/abs/1704.04861

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2 : Inverted Residuals and Linear Bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510–4520. https://doi.org/10.1109/CVPR.2018.00474

Abadi, M., Agarwal, A., Barham, P., *et al.* (2015). TensorFlow : Large-Scale Machine Learning on Heterogeneous Systems. Software disponible sur tensorflow.org.

Chollet, F., & autres (2015–). *Keras*. https://keras.io/

Ronacher, A. *et al.* (2010–). *Flask* (framework web Python). https://flask.palletsprojects.com/

---

*Dépôt GitHub : [Apple-leafs-classification](https://github.com/hmad-ait-lahmous/Apple-leafs-classification)*
