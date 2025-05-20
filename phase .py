import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from random import choice

# Sample customer data
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Age': [25, 45, 35, 50, 23],
    'AnnualIncome': [50000, 100000, 75000, 120000, 48000],
    'SpendingScore': [60, 30, 50, 20, 70]  # e.g., how much they spend
}

df = pd.DataFrame(data)

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'AnnualIncome', 'SpendingScore']])

# Segment customers using KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
df['Segment'] = kmeans.fit_predict(scaled_data)

# Personalized product offers by segment
product_offers = {
    0: ['Discount on luxury items', 'Early access to new products'],
    1: ['Cashback on essentials', 'Membership rewards'],
    2: ['Bundle offers', 'Limited-time deals']
}

# Personalized message generator
def generate_message(customer_row):
    name = f"Customer #{customer_row['CustomerID']}"
    segment = customer_row['Segment']
    offer = choice(product_offers[segment])
    return f"Hello {name}, we have a special offer for you: {offer}!"

# Generate personalized messages
df['MarketingMessage'] = df.apply(generate_message, axis=1)

print(df[['CustomerID', 'Segment', 'MarketingMessage']])
