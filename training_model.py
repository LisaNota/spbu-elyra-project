from feature_selection import full_prepared_train, full_prepared_test

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification



print("Работа в файле training_model.py")


X = full_prepared_train.drop('Transported', axis=1)
y = full_prepared_train['Transported'].astype(int)

cat_features = X.select_dtypes(exclude=['int', 'float']).columns.to_list()



print("\nОбучение модели CatBoost...")
print(X)
print(y)
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, cat_features=cat_features, verbose=0)

model.fit(X, y)

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'\nТочность на каждом dфолде: {scores}')
print(f'\nСредняя точность: {scores.mean()}')
print("\nОбучение модели завершено")
print("-"*30)



