import pandas as pd

# Load the CSV file
df = pd.read_csv("C:\\Users\\hp\\Desktop\\NLP_Proj\\Amazon_Reviews.csv")

# Display the shape and the first few rows
print("Shape of the dataset:", df.shape)
print(df.head())
