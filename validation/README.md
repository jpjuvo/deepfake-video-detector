# Second-level training

Second-level features were generated from first-level models and FaceNet embeddings. Second-level models were trained in a 5-fold cross-validation manner.

**Features**

- Mean, max, Q3, StD and median values of face predictions were calculated from each classifier model
- Max and mean consecutive FaceNet embedding distance
- Max and mean FaceNet embedding deviation from centroid embedding

**Models**

- XGB
- Logistic regression
- LightGBM