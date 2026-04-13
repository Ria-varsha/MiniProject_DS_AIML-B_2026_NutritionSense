import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_analysis(path):
    df = pd.read_csv(path)

    sns.set(style="whitegrid")

    # Map labels
    obesity_map = {
        0: "Insufficient Weight",
        1: "Normal Weight",
        2: "Overweight Level I",
        3: "Overweight Level II",
        4: "Obesity Type I",
        5: "Obesity Type II",
        6: "Obesity Type III"
    }

    df['Obesity_Label'] = df['NObeyesdad'].map(obesity_map)

    # Simplify categories
    def simplify(x):
        if x in ["Insufficient Weight", "Normal Weight"]:
            return "Normal"
        elif "Overweight" in x:
            return "Overweight"
        else:
            return "Obese"

    df['Obesity_Simple'] = df['Obesity_Label'].apply(simplify)

    # Plot graph
    plt.figure(figsize=(8,5))
    sns.countplot(x='Obesity_Simple', data=df)
    plt.title("Health Risk Distribution")
    plt.xlabel("Health Category")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    perform_analysis("../dataset/processed_data/cleaned_data.csv")