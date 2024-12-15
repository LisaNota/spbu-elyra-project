from training_model import model, full_prepared_test
import pandas as pd


print("Работа в файле make_predictions.py")
predictions = model.predict_proba(full_prepared_test)[:, 1]
pd.Series(predictions, name="Transported_Probability").to_csv('predictions.csv', index=False)

print("\nПайплайн завершен успешно! Файл с предсказаниями сохранен.")
