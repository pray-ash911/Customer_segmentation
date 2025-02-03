import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from Mall_Customers import df  # Ensure df is properly loaded

# Step 1: Select relevant features from the dataset
data = df[['Annual Income (k$)', 'Spending Score (1-100)']]  # Selecting the two columns for clustering

# Step 2: Standardize the data using StandardScaler (important for clustering algorithms)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)  # Scale the data so that both features have mean=0 and std=1
scaled_data = pd.DataFrame(scaled_data, columns=['Annual Income (k$)', 'Spending Score (1-100)'])  # Create DataFrame

# Step 3: Compute nearest neighbors to help find the optimal value of eps for DBSCAN
neighbors = NearestNeighbors(n_neighbors=10).fit(scaled_data)  # Fit the NearestNeighbors algorithm with 10 nearest neighbors
distances, indices = neighbors.kneighbors(scaled_data)  # Get distances to the 10 nearest neighbors for each point

# Step 4: Sort and plot the distances to visualize the "elbow" in the K-distance graph, which helps determine eps
distances = np.sort(distances[:, -1])  # Sort the distances of the 10th nearest neighbor for all points
plt.plot(distances)  # Plot the sorted distances
plt.title("K-distance Graph")  # Title of the plot
plt.xlabel("Points sorted by distance")  # Label for the x-axis
plt.ylabel("10th Nearest Neighbor Distance")  # Label for the y-axis
plt.show()  # Show the plot to help decide the optimal eps

# Step 5: Apply DBSCAN clustering algorithm
dbscan = DBSCAN(eps=0.5, min_samples=10)  # Initialize DBSCAN with eps=0.5 and min_samples=10
y_dbscan = dbscan.fit_predict(scaled_data[['Annual Income (k$)', 'Spending Score (1-100)']])  # Fit DBSCAN

# Step 6: Add the predicted DBSCAN clusters to the DataFrame for visualization
scaled_data['DBSCAN Cluster'] = y_dbscan  # Store the cluster labels in the DataFrame

# Step 7: Visualize the DBSCAN clustering result using a scatter plot
plt.figure(figsize=(8, 6))  # Set the figure size for the plot
sns.scatterplot(
    x='Annual Income (k$)',  # Set x-axis as Annual Income
    y='Spending Score (1-100)',  # Set y-axis as Spending Score
    hue='DBSCAN Cluster',  # Color the points based on the DBSCAN cluster
    data=scaled_data,  # Use the scaled_data DataFrame
    palette='viridis',  # Color palette for the clusters
    s=100  # Set the size of the scatter plot points
)
plt.title('DBSCAN Clustering')  # Set title for the plot
plt.xlabel('Annual Income (k$)')  # Label for x-axis
plt.ylabel('Spending Score (1-100)')  # Label for y-axis
plt.legend(title='Cluster')  # Add a legend to show cluster colors
plt.show()  # Show the scatter plot

# Step 8: Evaluate clustering performance using two evaluation metrics: Silhouette Score and Davies-Bouldin Index
# Exclude noise points (-1) when calculating evaluation scores
filtered_data = scaled_data[scaled_data['DBSCAN Cluster'] != -1]

silhouette = silhouette_score(filtered_data[['Annual Income (k$)', 'Spending Score (1-100)']], filtered_data['DBSCAN Cluster'])  # Silhouette Score
davies_bouldin = davies_bouldin_score(filtered_data[['Annual Income (k$)', 'Spending Score (1-100)']], filtered_data['DBSCAN Cluster'])  # Davies-Bouldin Index

# Step 9: Print the evaluation scores
print(f"DBSCAN Silhouette Score: {silhouette}")  # Higher is better
print(f"DBSCAN Davies-Bouldin Index: {davies_bouldin}")  # Lower is better
