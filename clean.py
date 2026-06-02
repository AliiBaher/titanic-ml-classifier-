import csv
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "titanic", "train.csv")
output_path = os.path.join(script_dir, "data", "titanic_clean.csv")

# Dosya satırları bazında ayrıştırma: bazı satırlar dıştaki tırnaklarla sarılmış
with open(input_path, "r", encoding="utf-8", newline="") as f:
    lines = [line.rstrip("\r\n") for line in f]

if not lines:
    raise ValueError(f"Boş dosya: {input_path}")

header = lines[0].split(",")
rows = []
for raw in lines[1:]:
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    raw = raw.replace('""', '"')
    rows.append(next(csv.reader([raw])))

df = pd.DataFrame(rows, columns=header)

# Weka uyumlu temizleme
# - gereksiz sütunları sil
# - eksik Age değerlerini ortalama ile doldur
# - eksik Embarked değerlerini 'S' ile doldur
# - nominal stringleri sayısal değerlere çevir
# - kalan NA satırları kaldır

keep_cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[keep_cols].copy()

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce")
df["Pclass"] = pd.to_numeric(df["Pclass"], errors="coerce")

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna("S")

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Eğer hâlâ eksik değer varsa bu satırları çıkar
df = df.dropna()

# Weka için tam sayıya çevrilebilir sütunlar
for col in ["Survived", "Pclass"]:
    df[col] = df[col].astype(int)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Weka için temiz CSV kaydedildi: {output_path}")
print(df.head())