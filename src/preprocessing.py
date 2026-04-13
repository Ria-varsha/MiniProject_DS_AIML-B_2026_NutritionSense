import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Remove missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Save processed data
    df.to_csv(output_path, index=False)

    print("Preprocessing completed and saved.")

    return df


if __name__ == "__main__":
    preprocess_data(
        "../dataset/raw_data/diet_health.csv",
        "../dataset/processed_data/cleaned_data.csv"
    )