## Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Load Dataset
df = pd.read_csv("train.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

## Explore Dataset
print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

## Data Cleaning
# Fill Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop Cabin
df.drop("Cabin", axis=1, inplace=True)

print("\nAfter Cleaning")
print(df.isnull().sum())

## Create Graph 1 – Survival Distribution
plt.figure(figsize=(6,4))

sns.countplot(x="Survived", data=df)

plt.title("Survival Distribution")

plt.savefig("graphs/survival_distribution.png")

plt.show()

## Create Graph 2 – Survival by Gender
plt.figure(figsize=(6,4))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Survival by Gender")

plt.savefig("graphs/survival_gender.png")

plt.show()

## Create Graph 3 – Passenger Class
plt.figure(figsize=(6,4))

sns.countplot(
    x="Pclass",
    data=df
)

plt.title("Passenger Class Distribution")

plt.savefig("graphs/passenger_class.png")

plt.show()

## Create Graph 4 – Survival by Class
plt.figure(figsize=(6,4))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df
)

plt.title("Survival by Passenger Class")

plt.savefig("graphs/survival_class.png")

plt.show()

## Create Graph 5 – Age Distribution
plt.figure(figsize=(8,5))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")

plt.savefig("graphs/age_distribution.png")

plt.show()

## Create Graph 6 – Fare Distribution
plt.figure(figsize=(8,5))

sns.histplot(
    df["Fare"],
    bins=30
)

plt.title("Fare Distribution")

plt.savefig("graphs/fare_distribution.png")

plt.show()

## Create Graph 7 – Correlation Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("graphs/heatmap.png")

plt.show()

print("\nEDA COMPLETED")

print("\nKey Findings:")

print("1. Females survived more than males.")

print("2. First-class passengers had the highest survival rate.")

print("3. Most passengers were aged between 20 and 40.")

print("4. Higher fare passengers had better survival chances.")

print("5. Third-class passengers had the highest death rate.")