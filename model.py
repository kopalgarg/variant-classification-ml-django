# array operations
import numpy as np
import pandas as pd
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# feature engineering and pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ml operations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

df = pd.read_csv('/Users/kopalgarg/Documents/GitHub/variant-classification-ml-django/clinvar_conflicting.csv', dtype={'CHROM': str, 38: str, 40: object})

keep = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP', 'CLNVC', 'CLASS','SIFT',
       'Allele', 'Consequence', 'IMPACT','CADD_PHRED', 'CADD_RAW']
df = df[keep]
df = df.dropna()

y = df['CLASS']
x = df.drop(columns=['CLASS'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10)

# pipeline
ohe_features = ['REF', 'ALT', 'CHROM', 'Allele', 'Consequence', 'SIFT','CLNVC', 'IMPACT']

categorical_preprocessing_ohe = Pipeline([('ohe', OneHotEncoder(handle_unknown = 'ignore'))])

preprocess = ColumnTransformer([
    ('categorical_preprocessing_ohe', categorical_preprocessing_ohe, ohe_features)
])

cls = Pipeline([
    ('preprocess', preprocess),
    ('clf', DecisionTreeClassifier())
])
cls.fit(x, y)
print("Model score: %.3f" % cls.score(x_test, y_test))
score = cls.score(x_test, y_test)
# use joblib to save the model
joblib.dump(cls, 'final_model.sav')
