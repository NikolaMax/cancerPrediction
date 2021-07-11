from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def makeModel(X_train, Y_train, X, Y):
    log = LogisticRegression(random_state=3, solver='liblinear')
    log.fit(X_train, Y_train)
    print('\n\n============================================================================')
    print('Cross validacija - logisticka regresija: ', cross_val_score(log, X, Y, cv=10, scoring='accuracy').mean())

    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)
    print('Cross validacija - decision tree: ', cross_val_score(tree, X, Y, cv=10, scoring='accuracy').mean())

    forest = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=7)
    forest.fit(X_train, Y_train)
    print('Cross validacija - random forest: ', cross_val_score(forest, X, Y, cv=10, scoring='accuracy').mean())

    print(' 0 - Tacnost Logisticke regresije (trening skupa):', log.score(X_train, Y_train))
    print(' 1 - Tacnost Decision Tree klasifikatora (trening skupa):', tree.score(X_train, Y_train))
    print(' 2 - Tacnost Random Forest klasifikatora (trening skupa):', forest.score(X_train, Y_train))
    print('============================================================================\n\n')

    return log, tree, forest
