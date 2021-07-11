import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cancerPrediction.utils import model

# Ucitavanje dataset-a
df = pd.read_csv('./data/data.csv')
# print(df.head())
# print(df.info())

# Broj kolona i redova u dataset-u
print(df.shape)

# Prebrojavanje null/empty vrednosti u svakoj koloni
print(df.isna().sum())

# Izbacivanje "prazne" kolone
df = df.dropna(axis=1)

# Nakon izbacivanja kolone (provera)
print(df.shape)

# Prebrojavanje malignih(M) i benignih(B) slucajeva
benignVsMalignant = df['diagnosis'].value_counts()
print(benignVsMalignant)

# Vizualizacija dijagnostikovanih slucajeva
# sns.countplot(x='diagnosis', data=df)
# plt.show()

# proveravanje tipova podataka iz kolona
# print(df.dtypes)

# konvertovanje string kolone (diagnosis) u brojeve 0/1
labelencoder_Y = LabelEncoder()
df.iloc[:, 1] = labelencoder_Y.fit_transform(df.iloc[:, 1].values)

# sns.pairplot(data=df.iloc[:,1:5], hue="diagnosis")
# plt.show()

# Prikazivanje korelacije izmedju kolona u dataset-u
# print(df.iloc[:, 1:32].corr())

# Vizualizacija korelacije
# plt.figure(figsize=(10, 10))
# sns.heatmap(df.iloc[:, 1:12].corr(), annot=True, fmt='.0%')
# plt.show()

# Podela dataset-a na nezavisne(x) i zavisne(y) dataset-ove
X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values

# Podela dataset-a na train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Skaliranje
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Pravljenje modela
model = model.makeModel(X_train, Y_train, X, Y)

print('============================================================================')
print('CONFUSION MATRIX')

# testiranje modela nad test podacima (confusion matrix)
for i in range(len(model)):
    print('Model ', i)
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    print(cm)
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    Accuracy = (TP + TN) / (TP + TN + FN + FP)
    print('Tacnost modela ', Accuracy)
    print('============================================================================\n')

print('ACCURACY SCORE')
# Drugi nacin racunanja metrike modela
for i in range(len(model)):
    print('Model ', i)
    print(classification_report(Y_test, model[i].predict(X_test)))
    print('Tacnost : ', accuracy_score(Y_test, model[i].predict(X_test)))
    print('============================================================================')

# Predikcija modela Random Forest klasifikatora
randomForest = model[2]
prediction = randomForest.predict(X_test)
print('Predikacija modela: ', prediction)
print()
print('Stvarni rezultat :   ', Y_test)
