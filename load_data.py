import pandas as pd
import numpy as np
import sys
sys.stdout = open('pipeline_output.log', 'w', encoding='utf-8')
sys.stderr = sys.stdout

print("-"*30)
print("Работа в файле load_data.py")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("\nНачало работы пайплайна")
print(f"\nРазмер тренировочного датасета: {train.shape}")
print(f"\nРазмер тестового датасета: {test.shape}")
print("-"*30)