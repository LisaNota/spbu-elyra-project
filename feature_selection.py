from processing_data import processed_train, processed_test
import pandas as pd
import sys


print("Работа в файле feature_selection.py")

def drop_unnecessary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Исключает лишние (неважные) признаки. Выбор основан на решении предыдущей домашней работы"""
    df = df.drop(['Name', 'Number', 'PassengerId', 'Group', 'HomePlanet', 'Destination', 'Cabin', 'VIP'], axis=1, errors='ignore')
    
    return df



print("\nУдаление признаков train")
full_prepared_train = drop_unnecessary_features(processed_train)
print("\nУдаление признаков test")
full_prepared_test = drop_unnecessary_features(processed_test)
print("\nУдаление признаков завершено")
print("-"*30)

# sys.stdout.close()