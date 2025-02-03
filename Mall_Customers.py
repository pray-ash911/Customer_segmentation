import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Loading the dataset from a CSV file
file_path = r"C:\Users\hp\Downloads\Mall_Customers\Mall_Customers.csv"
df = pd.read_csv(file_path)

# Display the first 5 rows of the dataset to get a quick overview of the data
print("First 5 rows of the dataset:")
print(df.head())

# Display information about the dataset (data types, number of non-null entries)
print("\nDataset Information:")
print(df.info())

# Check for missing values in the dataset (to identify if any data is missing)
print("\nMissing Values:")
print(df.isnull().sum())  # Calculate the number of missing values in each column

# Show summary statistics for numerical columns (mean, std, min, max, etc.)
print("\nSummary Statistics:")
print(df.describe())  # Provides measures like mean, std, min, max, etc.

# Visualize the distribution of the 'Age' column (can be useful for identifying data trends)
'''plt.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()'''

# Scatter plot to explore the relationship between 'Annual Income' and 'Spending Score'
'''plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])  # Using actual column names
plt.title("Income vs Spending Score")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()'''

# Extracting relevant columns (Annual Income and Spending Score) for segmentation
data = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Normalizing the data using MinMaxScaler to scale the features between 0 and 1
scaler = MinMaxScaler()

# Fit and transform the data (scaling 'Annual Income' and 'Spending Score')
scaled_data = scaler.fit_transform(data)

# Convert the scaled data back to a DataFrame to maintain column names
scaled_data = pd.DataFrame(scaled_data, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

# To determine the optimal number of clusters, we use the Elbow Method by calculating WCSS (Within-Cluster Sum of Squares)
wcss = []

# Try different K values (from 1 to 10) to find the optimal number of clusters
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)  # inertia_ represents WCSS

# Plotting the Elbow Method to visualize the WCSS for each value of K
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')  # The Elbow point indicates the optimal number of clusters
plt.show()

# Applying the K-Means algorithm with the optimal number of clusters (after elbow method)
optimal_k = 4  # Based on the Elbow method, we assume K=4 as the optimal number of clusters

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the model and predict the cluster labels for each data point
y_kmeans = kmeans.fit_predict(scaled_data)

# Adding the cluster labels to the scaled data for visualization and further analysis
scaled_data['Cluster'] = y_kmeans

# Checking the data with cluster labels added
scaled_data.head()

# Plotting the clusters with the assigned cluster labels
plt.scatter(scaled_data['Annual Income (k$)'], scaled_data['Spending Score (1-100)'], c=scaled_data['Cluster'], cmap='viridis')

# Adding the cluster centroids to the plot for better visualization
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', label='Centroids')

# Customizing the plot with title, labels, and legend
plt.title(f'K-Means Clustering (K={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Evaluation metrics for clustering
# The Silhouette Score indicates how well-defined the clusters are. A value closer to 1 indicates well-separated clusters.
sil_score = silhouette_score(scaled_data[['Annual Income (k$)', 'Spending Score (1-100)']], y_kmeans)
print(f"Silhouette Score: {sil_score}")

# The Davies-Bouldin Index (DBI) measures the average similarity ratio between clusters. A lower value indicates better clustering.
dbi_score = davies_bouldin_score(scaled_data[['Annual Income (k$)', 'Spending Score (1-100)']], y_kmeans)
print(f"Davies-Bouldin Index: {dbi_score}")

# Analyzing the means of the clusters to understand the customer segments better
cluster_means = scaled_data.groupby("Cluster").mean()

print("Mean values for each cluster:")
print(cluster_means)

'''
Step 2: Create a Strategy for Each Segment
Cluster	Description	Business Action
Cluster 0	Budget shoppers	Offer discounts or loyalty programs to increase spending.
Cluster 1	Luxury shoppers	Introduce premium or exclusive products.
Cluster 2	Balanced spenders	Advertise mid-tier products or upsell with minimal discounts.
Cluster 3	High income, low spending	Re-engagement campaigns, highlight premium offerings.
'''

# Save the clustered data to a new CSV file for further analysis or business use
scaled_data.to_csv('segmented_customers.csv', index=False)
print("Clustered data saved as 'segmented_customers.csv'")

# Optional visualization using Seaborn's scatterplot with hue representing clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=scaled_data,
    palette='viridis',
    s=100  # Adjusting size of points for visibility
)

# Customize the plot with title, labels, and legend
plt.title('Customer Segments with Labels')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
