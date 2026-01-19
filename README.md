# Human-Actions-Classes-From-Video

**Auteurs :** Alexis Moisy & Isidore Zongo

## Description du projet

Détection d'actions humaines dans des vidéos.

**Framework :** PyTorch + TorchVision  
**Datasets :** UCF-101, KTH Actions...  
**Approche :** Transfer learning d'un 3D-CNN sur Kinetics-400 (disponible dans TorchVision)

---

## Plan d'exploration approfondie : Détection d'actions humaines dans les vidéos

### Phase 1 : Fondamentaux théoriques (Semaine 1-2)

#### 1.1 Comprendre le problème
- **Définition** : Classification d'actions vs détection spatio-temporelle
- **Complexité** : Variabilité intra-classe, contexte, échelle temporelle
- **Applications** : Surveillance, analyse sportive, interfaces gestuelles, robotique

#### 1.2 Représentation vidéo
- **Frames vs séquences** : Pourquoi la dimension temporelle est cruciale
- **Optical flow** : Mouvement apparent entre frames
- **Spatio-temporal features** : Combinaison spatiale et temporelle

#### 1.3 Évolution des approches
- **Méthodes classiques** : HOG, SIFT, STIP
- **Deep Learning** : 2D-CNN + LSTM, 3D-CNN, Two-Stream, I3D
- **Architectures récentes** : SlowFast, X3D, Video Transformer

---

### Phase 2 : Datasets et leurs caractéristiques (Semaine 2-3)

#### 2.1 UCF-101
- **Caractéristiques** : 101 classes, ~13k vidéos, variabilité caméra/éclairage
- **Enjeux** : Diversité, bruit, annotations
- **Utilisation** : Benchmark, évaluation

#### 2.2 KTH Actions
- **Caractéristiques** : 6 actions, conditions contrôlées
- **Enjeux** : Simplicité vs généralisation
- **Utilisation** : Baseline, prototypage

#### 2.3 Kinetics-400
- **Caractéristiques** : 400 classes, ~300k clips YouTube, diversité élevée
- **Enjeux** : Qualité annotations, biais, échelle
- **Utilisation** : Pré-entraînement pour transfer learning

#### 2.4 Comparaison et choix
- **Variances** : Taille, diversité, qualité, licences
- **Critères de sélection** : Objectif, ressources, contraintes

---

### Phase 3 : Architectures 3D-CNN et transfer learning (Semaine 3-4)

#### 3.1 3D-CNN : principes
- **Convolutions 3D** : Spatial + temporel
- **Avantages** : Capture spatio-temporelle end-to-end
- **Limites** : Coût computationnel, mémoire

#### 3.2 Modèles pré-entraînés dans TorchVision
- **I3D (Inflated 3D)** : Inflation de 2D vers 3D
- **R3D, R(2+1)D** : Variantes ResNet 3D
- **MC3, MViT** : Architectures récentes
- **API TorchVision** : Chargement, fine-tuning

#### 3.3 Transfer learning
- **Stratégies** :
  - Feature extraction (gel des couches)
  - Fine-tuning (ajustement partiel)
  - End-to-end (réentraînement complet)
- **Choix** : Taille dataset, ressources, objectif

---

### Phase 4 : Enjeux et défis (Semaine 4-5)

#### 4.1 Enjeux techniques
- **Variabilité intra-classe** : Même action, exécutions différentes
- **Variabilité inter-classe** : Actions similaires
- **Contexte** : Objets, environnement, interactions
- **Échelle temporelle** : Actions courtes vs longues
- **Robustesse** : Éclairage, angle, résolution, bruit

#### 4.2 Enjeux computationnels
- **Mémoire** : Vidéos longues, batch size
- **Temps d'entraînement** : 3D-CNN coûteux
- **Optimisation** : Mixed precision, gradient checkpointing

#### 4.3 Enjeux méthodologiques
- **Overfitting** : Datasets limités
- **Généralisation** : Domain shift (YouTube vs surveillance)
- **Évaluation** : Métriques (accuracy, mAP), splits train/val/test
- **Reproductibilité** : Seeds, versions, environnement

---

### Phase 5 : Variances et limitations (Semaine 5-6)

#### 5.1 Variances à analyser
- **Variance d'exécution** : Vitesse, style personnel
- **Variance contextuelle** : Environnement, objets
- **Variance visuelle** : Caméra, éclairage, résolution
- **Variance temporelle** : Durée, rythme

#### 5.2 Limitations actuelles
- **Actions composées** : Séquences d'actions
- **Interactions** : Plusieurs personnes
- **Ambiguïté** : Actions similaires visuellement
- **Temps réel** : Latence, optimisation

#### 5.3 Biais et éthique
- **Biais de dataset** : Représentativité, diversité
- **Biais culturels** : Actions spécifiques à une culture
- **Privacy** : Surveillance, consentement

---

### Phase 6 : Implémentation pratique (Semaine 6-8)

#### 6.1 Setup environnement
```
Structure recommandée :
- data/
  - ucf101/
  - kth/
- models/
  - pretrained/
  - custom/
- notebooks/
  - exploration/
  - training/
- src/
  - data_loading.py
  - models.py
  - training.py
  - evaluation.py
```

#### 6.2 Exploration des données
- **Analyse statistique** : Distribution classes, durée vidéos
- **Visualisation** : Échantillons, optical flow
- **Préprocessing** : Découpage, normalisation, augmentation

#### 6.3 Implémentation baseline
- **Chargement données** : UCF-101 ou KTH
- **Modèle pré-entraîné** : I3D depuis TorchVision
- **Feature extraction** : Extraction de features
- **Classification** : Fine-tuning ou classifier linéaire

#### 6.4 Expérimentations
- **Ablation studies** : Impact des composants
- **Comparaisons** : Différents modèles pré-entraînés
- **Hyperparamètres** : Learning rate, batch size, augmentation
- **Évaluation** : Métriques, visualisation des erreurs

---

### Phase 7 : Documentation et synthèse (Semaine 8)

#### 7.1 Documentation technique
- **Architecture** : Schémas, choix de design
- **Résultats** : Tableaux, courbes, exemples
- **Code** : Commentaires, docstrings, README

#### 7.2 Synthèse théorique
- **Rapport** : État de l'art, méthodologie, résultats
- **Présentation** : Slides, démonstrations
- **Discussion** : Limites, améliorations, perspectives

---

## Ressources recommandées

### Papers essentiels
- "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" (I3D)
- "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (R(2+1)D)
- "SlowFast Networks for Video Recognition"

### Tutoriels pratiques
- PyTorch Video Tutorials
- TorchVision documentation (video models)
- Papers With Code (Action Recognition)

### Outils
- **TorchVision** : Modèles pré-entraînés
- **PyTorchVideo** : Bibliothèque vidéo Facebook
- **MMAction2** : Framework OpenMMLab
- **TensorFlow Hub** : Modèles alternatifs

---

## Métriques de progression

- [ ] Compréhension théorique des 3D-CNN
- [ ] Analyse comparative des datasets
- [ ] Implémentation d'un modèle pré-entraîné
- [ ] Fine-tuning sur UCF-101 ou KTH
- [ ] Analyse des erreurs et limitations
- [ ] Documentation complète

---

## Objectifs du projet

Le but est de pouvoir expliquer ce qu'est ce sujet :
- **Comment ?** Les variances, les enjeux, etc.
- **Approfondissement** : Compréhension en profondeur du domaine
- **Pratique** : Implémentation et expérimentation
