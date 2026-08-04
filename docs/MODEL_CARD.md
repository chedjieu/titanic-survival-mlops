# Model Card — titanic-survival-classifier

## Overview

Binary classifier predicting passenger survival on the Titanic using tabular features.

## Intended use

- Reference MLOps platform demonstration
- Educational / portfolio workloads
- Not for real-world life-critical decisions

## Training data

- Source: classic Titanic passenger CSV (`data/raw/titanic_data.csv`)
- Labels: `Survived` (0/1)
- Features engineered: travel group size, alone flag, encoded class/sex/embarked, scaled age/fare

## Model

- Algorithm: `RandomForestClassifier` inside an sklearn `Pipeline`
- Champion selected for production serving (BentoML + MLflow)
- Quality gate: F1 ≥ `MIN_F1_SCORE` (default 0.70)

## Metrics logged

Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importances

## Ethical considerations

Historical dataset reflects social biases of the era (sex, class). Do not generalize survival patterns to modern populations.
