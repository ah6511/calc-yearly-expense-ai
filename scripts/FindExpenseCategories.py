import os, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from typing import List, Optional, Tuple
from sklearn.preprocessing import normalize

import boto3

class FindExpenseCategories:
    def __init__(self):
        self.client = boto3.client(region_name="us-east-1",
                                service_name='bedrock-runtime',
                                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                                aws_session_token=os.environ["AWS_SESSION_TOKEN"])

    def dataframe_to_markdown(self, df):
        """
        Converts a Pandas DataFrame into a Markdown table.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be converted into a Markdown table.

        Returns
        -------
        str
            A string representing the DataFrame in Markdown table format.
            The table includes a header row with column names and rows
            formatted with proper Markdown table syntax."""
        
        # Create the header row
        markdown = f"| {' | '.join(df.columns)} |\n"
        markdown += f"| {' | '.join(['-' * len(col) for col in df.columns])} |\n"
        
        # Add each row
        for _, row in df.iterrows():
            markdown += f"| {' | '.join(str(cell) for cell in row)} |\n"
        
        return markdown

    def choose_best_k(self,
                    X: np.ndarray,
                    k_min: int,
                    k_max: int,
                    random_state: int = 42,
                    n_init: int = 10) -> Tuple[int, float]:
        """
        Returns (best_k, best_silhouette). Uses euclidean on L2-normalized embeddings (cosine-like).
        """
        best_k, best_score = None, -1.0
        for k in range(max(2, k_min), max(3, k_max + 1)):
            km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels, metric="euclidean")
            print(f"  k={k:>3} silhouette={score:.4f}")
            if score > best_score:
                best_k, best_score = k, score
        if best_k is None:
            raise RuntimeError("Failed to select best k.")
        print(f"Selected k={best_k} (silhouette={best_score:.4f})")
        return best_k, best_score

    def determine_clusters(self, df):
        # Standardizing data, so that all features have equal weight. This is important for modelling.
        # Otherwise, in our case Income would be considered much more important than Education for Instance. 
        # We do not know if this is the case, so we would not like to introduce it to our model. 
        # This is what is also referred to as bias.
        scaler = StandardScaler()
        df_std = scaler.fit_transform(df)

        # Perform K-means clustering. We consider 1 to 10 clusters, so our for loop runs 10 iterations.
        # In addition we run the algorithm at many different starting points - k means plus plus. 
        # And we set a random state for reproducibility.
        wcss = []
        silhouette_scores = []  # Silhouette scores for each k
        for i in range(2,20):
            kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans_pca.fit(df_std)
            wcss.append(kmeans_pca.inertia_)
            silhouette_scores.append(silhouette_score(df, kmeans_pca.labels_))

        if os.getenv("LOCAL_MODE", "false").lower() == "true":
            # Plot the Within Cluster Sum of Squares for the different number of clusters.
            # From this plot we choose the number of clusters. 
            # We look for a kink in the graphic, after which the descent of wcss isn't as pronounced.
            plt.figure(figsize = (10,8))
            plt.plot(range(2,20), wcss, marker = 'o', linestyle = '-.',color='red')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            plt.title('K-means Clustering')
            plt.show()

        # Determine the optimal number of clusters using the Elbow Method
        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []  # Silhouette scores for each k

        # Test different values of k
        k_values = range(2, 10)
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(df, kmeans.labels_))
        
        optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        return optimal_k

        # # Plot the Elbow Method
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.plot(k_values, wcss, 'bx-')
        # plt.xlabel('Number of clusters (k)')
        # plt.ylabel('WCSS')
        # plt.title('Elbow Method for Optimal k')

        # # Plot the Silhouette Scores
        # plt.subplot(1, 2, 2)
        # plt.plot(k_values, silhouette_scores, 'bx-')
        # plt.xlabel('Number of clusters (k)')
        # plt.ylabel('Silhouette Score')
        # plt.title('Silhouette Scores for Optimal k')

        # plt.tight_layout()
        # plt.show()

        # Choose the optimal k (e.g., based on the elbow or highest silhouette score)
        optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        print(f'Optimal number of clusters: {optimal_k}')
        return optimal_k

    def customer_segmentation(self, expenses):
        df = pd.DataFrame(expenses)
        
        # Vectorize transaction descriptions using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform( df['description'])

        n_clusters = self.determine_clusters(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        # Assign cluster labels to each transaction
        df['cluster'] = kmeans.labels_

        # Output the clustered data
        print(df)

        for i in range(n_clusters):
            transaction_data = df[df.cluster==i][['date', 'description', 'amount']]
            print("Segment", i+1, "| Transaction Data:", transaction_data)

    def get_embedding(self, text):
        # Create the request for the model.
        native_request = {"inputText": text}

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        # Invoke the model with the request.
        response = self.client.invoke_model(modelId=os.getenv('BEDROCK_EMBEDDING_MODEL_ID'), body=request)

        # Decode the model's native response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the generated embedding and the input text token count.
        embedding = model_response["embedding"]
        return embedding

    def cluster(self, expenses):
        df = pd.DataFrame(expenses)
        print(df.head())

        # Embed the expense descriptions
        embeddings = []
        for description in df['description'].tolist():
            embedding = self.get_embedding(description)
            embeddings.append(embedding)
            print(f"Description: {description}\nEmbedding: {embedding}\n")

        # best_k, _ = self.choose_best_k(embeddings, k_min=2, k_max=20, random_state=42)
        # k = best_k

        # # Perform PCA to reduce the dimensionality of the embeddings
        # pca = PCA(n_components=0.95)  # Retain 95% of the variance
        # reduced_embeddings = pca.fit_transform(np.array(embeddings))

        # # Perform K-Means clustering
        # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # df['cluster'] = kmeans.fit_predict(reduced_embeddings)

        # df1 = df.groupby('cluster').get_group(0)
        # # Calculate the average expense amount for each cluster
        # df['avg_expense'] = df.groupby('cluster')['amount'].transform('mean')

        # # Sort the clusters by average expense amount in descending order
        # df = df.sort_values('avg_expense', ascending=False)

        # # Get the top 5 expense descriptions
        # top_descriptions = df['description'].unique()[:10]

        # # Get the top 5 expense amounts
        # top_amounts = df['amount'].unique()[:10]

        # # Get the top 5 expense dates
        # top_dates = df['date'].unique()[:10]

        # # Get the top 5 expense vendor names
        # top_clusters = df['cluster'].unique()[:10]

        # # Get the top 5 expense vendor names
        # top_avg_expenses = df['avg_expense'].unique()[:10]

        # # Create a dictionary of the top 5 categories, descriptions, amounts, dates, vendors, clusters, and avg_expenses
        # top_dict = {'description': top_descriptions, 'amount': top_amounts, 'date': top_dates, 'cluster': top_clusters, 'avg_expense': top_avg_expenses}

        # # Create a dataframe of the top 5 categories, descriptions, amounts, dates, vendors, clusters, and avg_expenses
        # top_df = pd.DataFrame(top_dict)

        # # Return the top 5 categories, descriptions, amounts, dates, vendors, clusters, and avg_expenses
        # return top_df

        # create df_embeddings
        col_names = [f"description_embedding-{i+1}" for i in range(len(embeddings[0]))]
        df_embeddings = pd.DataFrame(embeddings, columns=col_names)
        print(df_embeddings.head())

        n_clusters = self.determine_clusters(df_embeddings)

        # KMeans
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(df_embeddings)

        # add segments to survey data
        df["segment"] = clustering.labels_+1

        # # use PCA to reduce dimensionality for visualization
        # X = PCA(n_components=2).fit_transform(df_embeddings)

        # # plot components for each segment
        # for label in range(n_clusters):
        #     indexes = np.where(clustering.labels_ == label)[0]
        #     plt.scatter(X[indexes,0], X[indexes,1], label=f"Segment {label+1}")

        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.xlabel("PC 1")
        # plt.ylabel("PC 2")
        # plt.show()

        for i in range(n_clusters):
            transaction_data = df[clustering.labels_==i][['date', 'description', 'amount']]
            print("Segment", i+1, "| Transaction Data:", transaction_data)
            # print(generate_avatar(my_sk, transaction_data))
            # print("-" * 100)

if __name__ == "__main__":
    findExpenseCategories = FindExpenseCategories()
    total_withdrawals_path = os.path.join(os.getcwd(), "transactions", "total_withdrawals.json")
    withdrawals = json.load(open(total_withdrawals_path, "r"))
    # findExpenseCategories.customer_segmentation(withdrawals)
    findExpenseCategories.cluster(withdrawals)