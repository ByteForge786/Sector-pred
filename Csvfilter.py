import pandas as pd

# Load the CSV into a DataFrame
df = pd.read_csv('your_csv_file.csv')

# Get unique values in the 'ratingsource' column
unique_labels = df['ratingsource'].unique()

# Create an empty DataFrame to store the result
result_df = pd.DataFrame(columns=df.columns)

# Iterate over unique labels and append 8 instances of each to the result DataFrame
for label in unique_labels:
    label_instances = df[df['ratingsource'] == label].head(8)
    result_df = pd.concat([result_df, label_instances], ignore_index=True)

# Output the result
print(result_df)
