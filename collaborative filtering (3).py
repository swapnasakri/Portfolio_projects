#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Model-based collaborative filtering
import numpy as np
import pandas as pd


# In[2]:


e_commerce=pd.read_csv('7817_1.csv')
e_commerce.head(1)


# In[3]:


print(e_commerce.info())


# In[4]:


print(e_commerce.head())


# In[5]:


e_commerce['reviews.username'].nunique()


# In[6]:


e_commerce['reviews.username'].nunique()


# In[7]:


e_commerce.isnull().sum()


# In[8]:


from sklearn.impute import SimpleImputer
# Fill missing values for categorical columns with their most frequent value (mode)
categorical_cols = ['dimension', 'colors', 'reviews.title', 'reviews.doRecommend', 'ean', 'manufacturer', 'manufacturerNumber', 'reviews.date', 'reviews.username', 'upc']
for col in categorical_cols:
    if col in e_commerce.columns:
        e_commerce[col] = e_commerce[col].fillna(e_commerce[col].mode()[0])

# Fill missing values for other columns with default values or appropriate strategies
e_commerce['reviews.userCity'] = e_commerce['reviews.userCity'].fillna(0)
e_commerce['reviews.userProvince'] = e_commerce['reviews.userProvince'].fillna(0)
e_commerce['sizes'] = e_commerce['sizes'].fillna(0)

# Convert numeric columns to numeric types
numeric_cols = ['reviews.numHelpful', 'reviews.rating', 'sizes', 'weight', 'prices']
for col in numeric_cols:
    if col in e_commerce.columns:
        e_commerce[col] = pd.to_numeric(e_commerce[col], errors='coerce')

# Identify columns that have all missing values
fully_missing_cols = [col for col in numeric_cols if e_commerce[col].isnull().all()]

# Handle columns with at least some non-missing values
imputer = SimpleImputer(strategy='mean')
non_missing_cols = [col for col in numeric_cols if col not in fully_missing_cols]
if non_missing_cols:
    e_commerce[non_missing_cols] = imputer.fit_transform(e_commerce[non_missing_cols])

# For fully missing columns, fill with a specific value (e.g., 0)
for col in fully_missing_cols:
    e_commerce[col] = e_commerce[col].fillna(0)


# In[9]:


e_commerce.isnull().sum()


# In[10]:


import matplotlib.pyplot as plt
# Calculate the distribution of ratings
# Extract the ratings column
rating = e_commerce['reviews.rating'].dropna()
rating_c = rating.value_counts().sort_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
rating_c.plot(kind='bar')
plt.title('Distribution of Kindle Review Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)

# Add value labels on top of each bar
for i, v in enumerate(rating_c):
    plt.text(i, v, str(v), ha='center', va='bottom')

#plt.tight_layout()
plt.show()

# Print summary statistics
print(rating_c)
print("\nTotal number of ratings:", rating.count())
print("Average rating:", rating.mean().round(2))
print("Median rating:", rating.median())


# In[11]:


#Text Preprocessing
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Setup lemmatizer and stopwords if they are available
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception:
    lemmatizer = None
    stop_words = set()

def Text_Preprocess(t):
    # Check if text is a string; if not, convert to string
    if not isinstance(t, str):
        t = str(t)
    
    # Remove special characters and punctuation
    t = re.sub(r'\W', ' ', t)
    
    # Convert to lowercase
    t = t.lower()
    
    # Tokenize by splitting on whitespace
    token = t.split()
    
    # Remove stop words if stop_words set is available
    if stop_words:
        token = [word for word in token if word not in stop_words]
    
    # Lemmatize if lemmatizer is available
    if lemmatizer:
        token = [lemmatizer.lemmatize(word) for word in token]
    
    return ' '.join(token)

# Apply the robust preprocessing function
e_commerce['processed_reviews.text'] = e_commerce['reviews.text'].apply(Text_Preprocess)
e_commerce['processed_reviews.title'] = e_commerce['reviews.title'].apply(Text_Preprocess)

# Display the first few rows of the cleaned columns alongside the original text
print(e_commerce[['reviews.text', 'processed_reviews.text', 'reviews.title', 'processed_reviews.title']].head())



# In[12]:


get_ipython().system('pip install textblob')
from collections import Counter
from textblob import TextBlob
import matplotlib.pyplot as plt
# Apply text cleaning to review text
e_commerce['cleaned_text'] = e_commerce['reviews.text'].apply(Text_Preprocess)

# Get most common words
count_word = [word for tokens in e_commerce['cleaned_text'] for word in tokens]
word_f = Counter(count_word)
common_words = word_f.most_common(10)

# Sentiment Analysis
e_commerce['sentiment'] = e_commerce['reviews.text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Visualize most common words
plt.figure(figsize=(12, 6))
plt.bar(*zip(*common_words))
plt.title('Top 10 Most Common Words in Reviews')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[13]:


# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
e_commerce['sentiment'].hist(bins=50)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Print summary statistics
print("Average Sentiment Score:", e_commerce['sentiment'].mean())
print("\nMost Common Words:")
for w, c in common_words:
    print(f"{w}: {c}")


# In[14]:


# Example of extracting positive and negative reviews
positive_review = e_commerce[e_commerce['sentiment'] > 0.5]['reviews.text'].head()
negative_review = e_commerce[e_commerce['sentiment'] < -0.5]['reviews.text'].head()

print("\nExample Positive Review:")
print(positive_review.iloc[0] if not positive_review.empty else "No strongly positive reviews found")

print("\nExample Negative Review:")
print(negative_review.iloc[0] if not negative_review.empty else "No strongly negative reviews found")


# In[15]:


#We can examine the temporal distribution of reviews to identify trends or changes in user sentiment over time
from datetime import datetime
e_commerce['reviews.date']=pd.to_datetime(e_commerce['reviews.date'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
#sort the date
e_commerce=e_commerce.sort_values('reviews.date')
#average rating
avg_rating=e_commerce.groupby('reviews.date')['reviews.rating'].mean().reset_index()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(avg_rating['reviews.date'], avg_rating['reviews.rating'])
plt.title('Average Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.ylim(0, 5)  
plt.grid(True)


# In[16]:


#user engagement
# User Engagement Analysis
user_review_count = e_commerce['reviews.username'].value_counts()
print("User Engagement:")
print(f"Total number of users: {len(user_review_count)}")
print(f"Top 10 most active users:")
print(user_review_count.head(10))
plt.figure(figsize=(12, 6))
user_review_count.head(20).plot(kind='bar')
plt.title('Top 20 Users by Number of Reviews')
plt.xlabel('Username')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[17]:


#Product popularity
product_count = e_commerce['name'].value_counts()
print(f"Total number of products: {len(product_count)}")
print(f"Top 10 most popular products:")
print(product_count.head(10))
plt.figure(figsize=(14, 6))
product_count.head(20).plot(kind='bar')
plt.title('Top 20 Users by Number of Reviews')
plt.xlabel('name')
plt.ylabel('Number of reviews')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Ensure 'prices' column is string type and handle missing values
e_commerce['prices'] = e_commerce['prices'].fillna('[]').astype(str)

# Convert prices from string to list of dictionaries
e_commerce['prices'] = e_commerce['prices'].apply(lambda x: json.loads(x) if x != '' else [])

# Extract the minimum price from each price list
e_commerce['min_price'] = e_commerce['prices'].apply(lambda x: min([float(p['amountMin']) for p in x if 'amountMin' in p]) if x else 0)

# Convert dimension string to numeric values
e_commerce['length'] = e_commerce['dimension'].str.extract('(\d+)').astype(float)

# Extract numeric rating
e_commerce['rating'] = e_commerce['reviews.rating'].astype(float)

# Ensure 'weight' column is string type before extracting numeric values
e_commerce['weight'] = e_commerce['weight'].astype(str)
e_commerce['weight_numeric'] = e_commerce['weight'].str.extract('(\d+)').astype(float)

# Select relevant columns for correlation
columns_for_correlation = ['min_price', 'length', 'rating', 'weight_numeric']

# Create correlation matrix
correlation_matrix = e_commerce[columns_for_correlation].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Kindle Paperwhite Features')
plt.tight_layout()
plt.show()


# In[19]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# Extract user-item interactions (ratings)
interactions = e_commerce[['reviews.username', 'asins', 'reviews.rating']]

# Drop rows with missing values
interactions = interactions.dropna()

# Convert ratings to numeric type
interactions['reviews.rating'] = pd.to_numeric(interactions['reviews.rating'], errors='coerce')

# Create a pivot table for user-item interactions
pivot_table = interactions.pivot_table(values='reviews.rating', index='reviews.username', columns='asins', fill_value=0)

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlOrRd', cbar_kws={'label': 'Rating'})
plt.title('User-Item Interaction Heatmap')
plt.xlabel('Item ASIN')
plt.ylabel('User')
plt.tight_layout()
plt.show()

# Print some statistics
print(f"Number of users: {pivot_table.shape[0]}")
print(f"Number of items: {pivot_table.shape[1]}")
print(f"Sparsity: {(pivot_table == 0).sum().sum() / (pivot_table.shape[0] * pivot_table.shape[1]):.2%}")


# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Extract user-item interactions (ratings)
interactions = e_commerce[['reviews.username', 'asins', 'reviews.rating']]

# Drop rows with missing values
interactions = interactions.dropna()

# Convert ratings to numeric type
interactions['reviews.rating'] = pd.to_numeric(interactions['reviews.rating'], errors='coerce')

# Create a pivot table for user-item interactions
pivot_table = interactions.pivot_table(values='reviews.rating', index='reviews.username', columns='asins', fill_value=0)

# Normalize the pivot table ratings
scaler = MinMaxScaler()
pivot_table_scaled = pd.DataFrame(scaler.fit_transform(pivot_table), index=pivot_table.index, columns=pivot_table.columns)

# Sort users and items by the number of interactions for better visibility
user_interactions = interactions['reviews.username'].value_counts()
item_interactions = interactions['asins'].value_counts()

# Select the top 30 users and items
top_users = user_interactions.head(30).index
top_items = item_interactions.head(30).index

# Filter the pivot table
pivot_table_filtered = pivot_table_scaled.loc[top_users, top_items]

# Create the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table_filtered, cmap='YlOrRd', cbar_kws={'label': 'Normalized Rating'}, annot=True, fmt=".2f")
plt.title('User-Item Interaction Heatmap (Top 30 Users and Items)')
plt.xlabel('Item ASIN')
plt.ylabel('User')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print some statistics
print(f"Number of users: {pivot_table.shape[0]}")
print(f"Number of items: {pivot_table.shape[1]}")
print(f"Sparsity: {(pivot_table == 0).sum().sum() / (pivot_table.shape[0] * pivot_table.shape[1]):.2%}")


# In[21]:


#Identify more potential influencers(users with morethan 5 reviews)
potential_influencers_count=user_review_count[user_review_count > 5]
print(f"Total number of potential influencers: {len(potential_influencers_count)}")
print("Top 10 potential influencers:")
print(potential_influencers_count.head(10))

# Calculate average reviews per user and per product
reviews_user = e_commerce['reviews.username'].value_counts().mean()
reviews_product = e_commerce['name'].value_counts().mean()

print(f"\nAverage reviews per user: {reviews_user:.2f}")
print(f"Average reviews per product: {reviews_product:.2f}")


# In[22]:


# Convert 'reviews.username' from strings to categorical codes
e_commerce['user_id'] = e_commerce['reviews.username'].astype('category').cat.codes
print(e_commerce[['reviews.username', 'user_id']].head())
user_id=e_commerce['user_id']


# In[23]:


#Convert item identifiers to strings
e_commerce['item_id'] = e_commerce['asins'].astype(str)
e_commerce['item_id']=e_commerce['item_id'].astype('category').cat.codes

item_id=e_commerce['item_id']
e_commerce[['asins','item_id']].head(25)


# In[24]:


from scipy.sparse import csr_matrix
#create a sparse matrix
user_rate_matrix=csr_matrix(
                  (e_commerce['reviews.rating'],(user_id,item_id)),
                     shape=(user_id.max()+1, item_id.max()+1))
print('shape of the matrix',user_rate_matrix.shape)


# In[25]:


from scipy.sparse.linalg import svds
import numpy as np

l,d, v=svds(user_rate_matrix, k=50)
si=np.diag(d)


# In[26]:


pred_rating=np.dot(np.dot(l, si),v)


# In[27]:


def recommend_items(user, matrix, num_recommendations=5):
    user_ratings = matrix[user]
    recommendations = np.argsort(user_ratings)[-num_recommendations:][::-1]
    return recommendations


user = 151 
recommended_items = recommend_items(user, pred_rating)
print("Recommended Items for User ", user, ":", recommended_items)
    


# In[ ]:





# In[28]:


get_ipython().system('pip install scikit-surprise')


# In[29]:


try:
    from surprise import Dataset, Reader
    from surprise import SVD, NMF, SlopeOne, KNNBasic
    from surprise.model_selection import cross_validate
    print("Surprise library successfully imported")
except ImportError as e:
    print(f"Error importing Surprise: {e}")
    print("Please make sure scikit-surprise is installed correctly")


# In[30]:


import surprise
print(surprise.__version__)


# In[31]:


import numpy as np

def ndcg_at_k(actual, predicted, k):
    dcg_k = 0
    idcg_k = 0
    
    for i in range(min(k, len(actual))):
        if predicted[i] in actual:
            dcg_k += 1 / np.log2(i + 2)
        idcg_k += 1 / np.log2(i + 2)
    
    return dcg_k / idcg_k if idcg_k > 0 else 0

def hit_rate_at_k(actual, predicted, k):
    return int(len(set(actual) & set(predicted[:k])) > 0)


# In[32]:


from surprise import Dataset, Reader
from surprise import SVD, NMF, SlopeOne, KNNBasic
from surprise.model_selection import cross_validate
#from surprise.accuracy import ndcg_at_k, hit_rate_at_k
from collections import defaultdict

# Select relevant columns
ratings_e = e_commerce[['reviews.username', 'asins', 'reviews.rating']]

# Create a Surprise dataset
reader = Reader(rating_scale=(1, 5))
d = Dataset.load_from_df(ratings_e, reader)

# Define the algorithms to compare
algorithms = {
    "SVD": SVD(),
    "NMF": NMF(),
    "SlopeOne": SlopeOne(),
    "KNN": KNNBasic()
}

# Perform cross-validation
results = {}
for n, a in algorithms.items():
    results[n] = cross_validate(a, d, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print results
for n, r in results.items():
    print(f"\n{n}:")
    print(f"RMSE: {r['test_rmse'].mean():.3f} (+/- {r['test_rmse'].std():.3f})")
    print(f"MAE: {r['test_mae'].mean():.3f} (+/- {r['test_mae'].std():.3f})")

# Function to get top-N recommendations
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Function to compute NDCG@k and Hit Rate@k
def compute_metrics(predictions, k=10):
    top_n = get_top_n(predictions, n=k)
    ndcg_sum = 0
    hit_sum = 0
    num_users = len(top_n)
    
    for uid, user_ratings in top_n.items():
        user_true_ratings = [pred.r_ui for pred in predictions if pred.uid == uid]
        ideal_dcg = np.sum((2**np.sort(user_true_ratings)[::-1] - 1) / np.log2(np.arange(2, len(user_true_ratings) + 2)))
        
        dcg = 0
        hit = 0
        for i, (iid, _) in enumerate(user_ratings):
            true_rating = next((pred.r_ui for pred in predictions if pred.uid == uid and pred.iid == iid), 0)
            if true_rating > 0:
                dcg += (2**true_rating - 1) / np.log2(i + 2)
                hit = 1
                break
        
        ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 0
        hit_sum += hit
    
    ndcg = ndcg_sum / num_users
    hit_rate = hit_sum / num_users
    
    return ndcg, hit_rate

# Compute NDCG@10 and Hit Rate@10 for each algorithm
for name, algorithm in algorithms.items():
    trainset = d.build_full_trainset()
    algorithm.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = algorithm.test(testset)
    ndcg, hit_rate = compute_metrics(predictions, k=10)
    print(f"\n{name}:")
    print(f"NDCG@10: {ndcg:.3f}")
    print(f"Hit Rate@10: {hit_rate:.3f}")
    
    


# In[36]:


import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader
from surprise import SVD, NMF, SlopeOne, KNNBasic
from surprise.model_selection import cross_validate
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Select relevant columns and drop NaN values
ratings_e = e_commerce[['reviews.username', 'asins', 'reviews.rating']].dropna()

# Create a Surprise dataset
reader = Reader(rating_scale=(1, 5))
d = Dataset.load_from_df(ratings_e, reader)

# Define the algorithms to compare
algorithms = {
    "SVD": SVD(),
    "NMF": NMF(),
    "SlopeOne": SlopeOne(),
    "KNN": KNNBasic()
}

# Perform cross-validation
results = {}
for n, a in algorithms.items():
    results[n] = cross_validate(a, d, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print results
for n, r in results.items():
    print(f"\n{n}:")
    print(f"RMSE: {r['test_rmse'].mean():.3f} (+/- {r['test_rmse'].std():.3f})")
    print(f"MAE: {r['test_mae'].mean():.3f} (+/- {r['test_mae'].std():.3f})")

# Function to get top-N recommendations
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Function to compute NDCG@k and Hit Rate@k
def compute_metrics(predictions, k=10):
    top_n = get_top_n(predictions, n=k)
    ndcg_sum = 0
    hit_sum = 0
    num_users = len(top_n)
    
    for uid, user_ratings in top_n.items():
        user_true_ratings = [pred.r_ui for pred in predictions if pred.uid == uid]
        ideal_dcg = np.sum((2**np.sort(user_true_ratings)[::-1] - 1) / np.log2(np.arange(2, len(user_true_ratings) + 2)))
        
        dcg = 0
        hit = 0
        for i, (iid, _) in enumerate(user_ratings):
            true_rating = next((pred.r_ui for pred in predictions if pred.uid == uid and pred.iid == iid), 0)
            if true_rating > 0:
                dcg += (2**true_rating - 1) / np.log2(i + 2)
                hit = 1
                break
        
        ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 0
        hit_sum += hit
    
    ndcg = ndcg_sum / num_users
    hit_rate = hit_sum / num_users
    
    return ndcg, hit_rate

# Compute NDCG@10 and Hit Rate@10 for each algorithm
ndcg_scores = {}
hit_rates = {}
for name, algorithm in algorithms.items():
    trainset = d.build_full_trainset()
    algorithm.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = algorithm.test(testset)
    ndcg, hit_rate = compute_metrics(predictions, k=10)
    ndcg_scores[name] = ndcg
    hit_rates[name] = hit_rate
    print(f"\n{name}:")
    print(f"NDCG@10: {ndcg:.3f}")
    print(f"Hit Rate@10: {hit_rate:.3f}")

# Plot RMSE and MAE
algorithms_names = list(algorithms.keys())
rmse_scores = [results[algo]['test_rmse'].mean() for algo in algorithms_names]
mae_scores = [results[algo]['test_mae'].mean() for algo in algorithms_names]

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.bar(algorithms_names, rmse_scores, color='skyblue')
plt.xlabel('Algorithms')
plt.ylabel('RMSE')
plt.title('RMSE for Different Algorithms')

plt.subplot(1, 2, 2)
plt.bar(algorithms_names, mae_scores, color='lightgreen')
plt.xlabel('Algorithms')
plt.ylabel('MAE')
plt.title('MAE for Different Algorithms')

plt.tight_layout()
plt.show()

# Plot NDCG@10 and Hit Rate@10
ndcg_values = list(ndcg_scores.values())
hit_rate_values = list(hit_rates.values())

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.bar(algorithms_names, ndcg_values, color='coral')
plt.xlabel('Algorithms')
plt.ylabel('NDCG@10')
plt.title('NDCG@10 for Different Algorithms')

plt.subplot(1, 2, 2)
plt.bar(algorithms_names, hit_rate_values, color='orchid')
plt.xlabel('Algorithms')
plt.ylabel('Hit Rate@10')
plt.title('Hit Rate@10 for Different Algorithms')

plt.tight_layout()
plt.show()



# In[34]:


# Train the SVD algorithm on the entire dataset
trainset = d.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Function to get top-N recommendations
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n
# Function to get recommendations for a product
def get_product_recommendations(product_id, n=10):
    # Get all users who have rated this product
    users_rated = e_commerce[e_commerce['asins'] == product_id]['reviews.username'].unique()
    
    # Get all products these users have rated
    related_products = e_commerce[e_commerce['reviews.username'].isin(users_rated)]['asins'].unique()
    
    # Make predictions for all related products
    testset = [(user, prod, 4.) for user in users_rated for prod in related_products if prod != product_id]
    pred = algo.test(testset)
    
    # Get top-N recommendations
    top_n = get_top_n(predictions, n)
    
    # Aggregate recommendations across users
    product_scores = defaultdict(list)
    for user in top_n:
        for prod, score in top_n[user]:
            product_scores[prod].append(score)
    
    # Calculate average score for each product
    avg_scores = [(prod, sum(scores)/len(scores)) for prod, scores in product_scores.items()]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    
    return avg_scores[:n]

# Get recommendations for a specific product (replace 'B00QJDU3KY' with an actual product ID from your dataset)
product_id = 'B00QJDU3KY'
recommendations = get_product_recommendations(product_id, n=10)

print(f"\nTop 10 recommendations for products similar to {product_id}:")
for i, (item_id, estimated_rating) in enumerate(recommendations, 1):
    print(f"{i}. Product ID: {item_id}, Estimated Rating: {estimated_rating:.2f}")


# In[35]:


#Clustering 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
e_commerce = pd.read_csv('7817_1.csv')

# Preprocess the data
e_commerce['price'] = e_commerce['prices'].apply(lambda x: eval(x)[0]['amountMin'] if x else np.nan)
e_commerce['rating'] = e_commerce['reviews.rating'].fillna(0)

# Create feature matrix
feature_matrix = ['price', 'rating']
X = e_commerce[feature_matrix].fillna(0)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform text vectorization on product names
tfidf = TfidfVectorizer(stop_words='english')
text_features = tfidf.fit_transform(e_commerce['name'].fillna(''))

# Combine numerical and text features
X_mixed = np.hstack((X_scaled, text_features.toarray()))

# Perform K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
e_commerce['cluster'] = kmeans.fit_predict(X_mixed)

# Function to get recommendations based on an ASIN
def get_recommendations(asin, top_n=5):
    product = e_commerce[e_commerce['asins'] == asin].iloc[0]
    cluster = product['cluster']
    
    # Get other products in the same cluster
    cluster_prod = e_commerce[e_commerce['cluster'] == cluster]
    
    # Calculate similarity within the cluster
    cluster_feature = X_mixed[e_commerce['cluster'] == cluster]
    similarity = cosine_similarity(cluster_feature)
    
    # Get the index of the input product within its cluster
    product_i = cluster_prod.index.get_loc(product.name)
    
    # Sort similar products
    s_index = similarity[product_i].argsort()[::-1]
    similar_product = cluster_prod.iloc[s_index]
    
    # Remove the input product and return top N recommendations
    rec = similar_product[similar_product['asins'] != asin].head(top_n)
    
    return rec[['asins', 'name', 'price', 'rating']]

# Example usage with a specific ASIN
asin_id = 'B00QJDU3KY'  # This is the ASIN for Kindle Paperwhite

print(f"Product details for ASIN {asin_id}:")
product = e_commerce[e_commerce['asins'] == asin_id].iloc[0]
print(f"Name: {product['name']}")
print(f"Price: ${product['price']:.2f}")
print(f"Rating: {product['rating']:.1f}")

print("\nRecommendations:")
recommendations = get_recommendations(asin_id)
for _, rec in recommendations.iterrows():
    print(f"- {rec['name']} (ASIN: {rec['asins']})")
    print(f"  Price: ${rec['price']:.2f}, Rating: {rec['rating']:.1f}")
    print()

# Print cluster statistics
print("\nCluster Statistics:")
for i in range(n_clusters):
    cluster_size = (e_commerce['cluster'] == i).sum()
    avg_price = e_commerce[e_commerce['cluster'] == i]['price'].mean()
    avg_rating = e_commerce[e_commerce['cluster'] == i]['rating'].mean()
    print(f"Cluster {i}: Size = {cluster_size}, Avg Price = ${avg_price:.2f}, Avg Rating = {avg_rating:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




