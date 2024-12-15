import pandas as pd
from load_data import train, test

print("Работа в файле processing_data.py")


def process_data(df:pd.DataFrame) -> pd.DataFrame:
    """Подготавливает датасет: заполняет пропуски, создает новые признаки.
    Взято из предыдущей работы"""
    
    df = df.copy()
    print(f"\nОбщее количество пропусков {df.isna().sum().sum()}")
    
    
    service_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    mean_age = int(df['Age'].mean())
    
    df[service_features] = df[service_features].fillna(0)
    df['Age'] = df['Age'].fillna(mean_age)
    
    print(f"\nЗаполнены пропуски в столбцах {service_features}")
    
    df['CryoSleep'] = df['CryoSleep'].apply(lambda x: 1 if x else 0)
    df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
    
    df[['Deck', 'Number', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Side'] = df['Side'].apply(lambda x: 1 if x == 'P' else 0)
    df['Deck'], _ = df['Deck'].factorize()

    known_planets = ['Earth', 'Europa', 'Mars']

    for planet in known_planets:
        df[f"Planet_{planet}"] = 0
        df.loc[df['HomePlanet'] == planet, f'Planet_{planet}'] = 1


    known_Destinations = ['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']
    for dist in known_Destinations:
        df[f"Destination_{dist}"] = 0
        df.loc[df['Destination'] == dist, f'Destination_{dist}'] = 1
        
        
    print(f"\nКатегориальные признаки разбиты с помощью one hot encoding")
    
    df['total_expenses'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa']

    df[['Group', 'NumberOfGroup']] = df['PassengerId'].str.split('_', expand=True)
    df[['Group', 'NumberOfGroup']] = df[['Group', 'NumberOfGroup']].astype(int)
    df['GroupSize'] = df.groupby('Group')['NumberOfGroup'].transform('sum')

    
    print(f"\nСгенерированы новые признаки: total_expenses, group, number_of_group")
    
    return df



print("\nОбработка train")
processed_train = process_data(train)
print("\nОбработка test")
processed_test = process_data(test)                      
print("\nПрепроцессинг данных завершен")
print("-"*30)