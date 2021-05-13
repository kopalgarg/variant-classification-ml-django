# array operations
import numpy as np
import pandas as pd
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# feature engineering
from sklearn.feature_extraction import FeatureHasher
# ml operations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

# columns (0,38,40) have mixed types so specifying dtype option while importing
df = pd.read_csv('/Users/kopalgarg/Documents/GitHub/variant-classification-ml-django/clinvar_conflicting.csv', dtype={'CHROM': str, 38: str, 40: object})

# remove null values, and cols with < 1000 unique values
keep = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP',
       'CLNDISDB', 'CLNDN', 'CLNHGVS', 'CLNVC', 'ORIGIN', 'CLASS',
       'Allele', 'Consequence', 'IMPACT', 'SYMBOL',
       'Feature', 'BIOTYPE', 'STRAND','CADD_PHRED', 'CADD_RAW']
df = df[keep]
remove = []
for i in df.columns.values:
    if df[i].nunique() < 1000:
        remove.append(i)
df = df[remove]
df['CHROM'] = df['CHROM'].astype(str)

# feature hashing and one-hot encoding (to deal with string variables)
hasher = FeatureHasher(n_features = 5, input_type = 'string')
REF = pd.DataFrame(hasher.fit_transform(df['REF']).toarray())
nameList = {}
for i in REF.columns.values:
    nameList[i] = "REF"+str(i+1)
REF.rename(columns = nameList, inplace = True)

ALT = pd.DataFrame(hasher.fit_transform(df['ALT']).toarray())
nameList = {}
for i in ALT.columns.values:
    nameList[i] = "ALT"+str(i+1)
ALT.rename(columns = nameList, inplace = True)

CHROM = pd.DataFrame(hasher.fit_transform(df['CHROM']).toarray())
nameList = {}
for i in CHROM.columns.values:
    nameList[i] = "CHROM"+str(i+1)
CHROM.rename(columns = nameList, inplace = True)

Allele = pd.DataFrame(hasher.fit_transform(df['Allele']).toarray())
nameList = {}
for i in Allele.columns.values:
    nameList[i] = "Allele"+str(i+1)
Allele.rename(columns = nameList, inplace = True)


Consequence = pd.DataFrame(hasher.fit_transform(df['Consequence']).toarray())
nameList = {}
for i in Consequence.columns.values:
    nameList[i] = "Consequence"+str(i+1)
Consequence.rename(columns = nameList, inplace = True)

# one hot encoding
CLNVC = pd.get_dummies(df['CLNVC'])
IMPACT = pd.get_dummies(df['IMPACT'])
BIOTYPE = pd.get_dummies(df['BIOTYPE'], drop_first=True)
STRAND = pd.get_dummies(df['STRAND'], drop_first=True)

df = pd.concat([REF, ALT, CHROM, Allele, Consequence , CLNVC, IMPACT, BIOTYPE,STRAND, df['CLASS']], axis=1)
df = df.dropna()
df.rename(columns={1 : "one"}, inplace = True)


y = df['CLASS']
x = df.drop(columns=['CLASS'], axis = 1)
df.head() # 65188 data points and 29 cols

# split into train-test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# decision tree model
dtree = DecisionTreeClassifier(max_depth = 5)
dtree.fit(x_train, y_train)
y_test_pred = dtree.predict(x_test)

print(accuracy_score(y_test, y_test_pred))

# use joblib to save the model
joblib.dump(dtree, 'final_model.sav')
