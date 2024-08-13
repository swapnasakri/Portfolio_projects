#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


e_commerce=pd.read_csv('7817_1.csv')
e_commerce.head(1)


# In[3]:


e_commerce.shape


# In[4]:


#asins: Unique identifier for the product which can be used to link different data points.
#brand: The brand of the product.
#categories: Categories under which the product is listed.
#dimension: Physical dimensions of the product.
#name: Name of the product.
#colors: Available colors for the product.
#weight: Weight of the product.
#Textual Data (for NLP processing)
#reviews.text: Text of the product reviews, which can provide insights into the product's features and user opinions.
#reviews.title: Title of the reviews, which can also provide summarized insights.
#User Feedback and Ratings
#reviews.rating: Ratings given by users, which indicate the level of user satisfaction.
#reviews.doRecommend: Whether users recommend the product, providing a binary indication of satisfaction
e_commerce=e_commerce[['asins','brand','categories','dimension','name','colors','weight','prices','reviews.text','reviews.title','reviews.rating','reviews.doRecommend']]
e_commerce


# In[5]:


e_commerce.info()


# In[6]:


e_commerce.isnull().sum()


# In[7]:


# Fill missing  values with the most frequent value
e_commerce['dimension'] = e_commerce['dimension'].fillna(e_commerce['dimension'].mode()[0])
e_commerce['colors'] = e_commerce['colors'].fillna(e_commerce['colors'].mode()[0])
# Filling missing values in 'weight' column with its mode
e_commerce['weight'] = e_commerce['weight'].fillna(e_commerce['weight'].mode()[0])
# Filling missing values in 'reviews.rating' column with its mean
e_commerce['reviews.rating'] = e_commerce['reviews.rating'].fillna(e_commerce['reviews.rating'].mean())
# Filling missing values in 'reviews.title' column with its mode
e_commerce['reviews.title'] = e_commerce['reviews.title'].fillna(e_commerce['reviews.title'].mode()[0])
# Filling missing values in 'reviews.doRecommend' column with its mode
e_commerce['reviews.doRecommend'] = e_commerce['reviews.doRecommend'].fillna(e_commerce['reviews.doRecommend'].mode()[0])


# In[8]:


e_commerce.isnull().sum()


# In[9]:


e_commerce.duplicated().sum()


# In[10]:


# Remove duplicate rows
e_commerce= e_commerce.drop_duplicates()


# In[11]:


e_commerce.duplicated().sum()


# In[12]:


e_commerce['prices'].value_counts()


# In[13]:


import ast
def convert (j):
    L=[]
    for i in ast.literal_eval(j):
        L.append(i['amountMax'])
        L.append(i['amountMin'])
        return L


# In[14]:


e_commerce['prices']=e_commerce['prices'].apply(convert)


# In[15]:


e_commerce.head()


# In[16]:


#Data analysis 
#1.  Distribution of Ratings
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.countplot(x='reviews.rating', data=e_commerce, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()


# In[17]:


####### not added ................# 2. Rating Distribution Across Categories
mean_ratings = e_commerce.groupby('categories')['reviews.rating'].mean().nlargest(20)

# Plot mean ratings for top categories
plt.figure(figsize=(12, 8))
sns.barplot(y=mean_ratings.index, x=mean_ratings.values, palette='viridis')
plt.title('Mean Ratings for Top 20 Categories')
plt.xlabel('Categories')
plt.ylabel('Mean Rating')
plt.xticks(rotation=45)
plt.show()


# In[18]:


#3. Length Of Reviews.
# Calculate the length of each review
e_commerce['review_length'] = e_commerce['reviews.text'].apply(lambda x: len(x.split()))

# Plot the distribution of review lengths
plt.figure(figsize=(10, 6))
sns.histplot(e_commerce['review_length'], bins=25, kde=True, color='blue')
plt.title('Length Of Reviews')
plt.xlabel('Number of Words in Review')
plt.ylabel('Frequency')
plt.show()


# In[19]:


#Frequency Categories
plt.figure(figsize=(16, 10))
top_categories = e_commerce['categories'].value_counts().nlargest(25)
sns.barplot(y=top_categories.index, x=top_categories.values)
plt.title('Top 25 Categories')
plt.xlabel('Count')
plt.ylabel('Categories')
plt.xticks(rotation=50)
plt.show()


# In[20]:


# Convert text columns to strings
text_columns = ['name', 'categories', 'brand', 'dimension', 'colors', 'reviews.text', 'reviews.title']
for c in text_columns:
    e_commerce[c] = e_commerce[c].astype(str)

# Convert numerical columns to numeric types
num_columns = ['reviews.rating', 'reviews.doRecommend']
for cl in num_columns:
    e_commerce[cl] = pd.to_numeric(e_commerce[cl], errors='coerce')

# Fill missing values
e_commerce.fillna('', inplace=True)


# In[21]:


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


# In[22]:


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


# In[23]:


# Visualize sentiment distribution
plt.figure(figsize=(9, 6))
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


# In[24]:


# Example of extracting positive and negative reviews
positive_review = e_commerce[e_commerce['sentiment'] > 0.5]['reviews.text'].head()
negative_review = e_commerce[e_commerce['sentiment'] < -0.5]['reviews.text'].head()

print("\nExample Positive Review:")
print(positive_review.iloc[0] if not positive_review.empty else "No strongly positive reviews found")

print("\nExample Negative Review:")
print(negative_review.iloc[0] if not negative_review.empty else "No strongly negative reviews found")


# In[25]:


e_commerce.head()


# In[26]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
# Combine relevant features into a single feature
e_commerce['combined_features'] = (
    e_commerce['name'].astype(str) + ' ' +
    e_commerce['categories'].astype(str) + ' ' +
    e_commerce['brand'].astype(str) + ' ' +
    e_commerce['dimension'].astype(str) + ' ' +
    e_commerce['colors'].astype(str) + ' ' +
    e_commerce['reviews.text'].astype(str) + ' ' +
    e_commerce['reviews.title'].astype(str) + ' ' +
    e_commerce['reviews.rating'].astype(str) + ' ' +
    e_commerce['reviews.doRecommend'].astype(str)
)

# Clean and preprocess the data if necessary
e_commerce['combined_features'] = e_commerce['combined_features'].fillna('')  # Filling NaN values


# In[27]:


e_commerce.head()


# In[28]:


# Step 2: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(e_commerce['combined_features'])


# In[29]:


from sklearn.metrics.pairwise import cosine_similarity

# Step 3: Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a DataFrame for the similarity matrix
valid_asins = e_commerce['asins'].values
cosine_sim_df = pd.DataFrame(cosine_sim, index=valid_asins, columns=valid_asins)



# In[30]:


# Building the recommender function
def get_recommendations(title, e_commerce, cosine_sim):
    
    # Find the index of the product that matches the product name
    if title not in e_commerce['name'].values:
        return "Product not found. Please check the name."

    # Get the index of the product that matches the title
    idx = e_commerce[e_commerce['name'] == title].index[0]

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar products
    top_indices = [i[0] for i in sim_scores[1:11]]  # skip self-match at index 0
    return e_commerce.iloc[top_indices]
# Example usage
recommendations = get_recommendations('Kindle Fire HD 7"', e_commerce, cosine_sim)
if isinstance(recommendations, pd.DataFrame):
    print(recommendations[['asins', 'name', 'categories', 'prices', 'colors', 'reviews.rating', 'reviews.text']])
else:
    print(recommendations)  # This will show an error message if the product is not found


# In[31]:


# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix_train, tfidf_matrix_train)

# Building the recommender function
def get_recommendations(title, train, cosine_sim):
    if title not in train['name'].values:
        return "Product not found. Please check the name."

    idx = train[train['name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:11]]
    return train.iloc[top_indices]

# Define evaluation metrics
def precision_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / k

def recall_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / len(y_true)

# Evaluate the model
def evaluate_model(test_set, train_set, cosine_sim, num_recommendations=5):
    precision_scores = []
    recall_scores = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        product_name = test_set.iloc[idx]['name']
        recommendations = get_recommendations(product_name, train_set, cosine_sim)
        if isinstance(recommendations, str):  # if the product is not found
            continue
        recommended_product_ids = recommendations['asins'].values

        y_true = [true_product_id]
        y_pred = recommended_product_ids.tolist()

        precision_scores.append(precision_at_k(y_true, y_pred, num_recommendations))
        recall_scores.append(recall_at_k(y_true, y_pred, num_recommendations))

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    return avg_precision, avg_recall

# Example usage for a single product
recommendations = get_recommendations('Kindle Fire HD 7"', train, cosine_sim)
if isinstance(recommendations, pd.DataFrame):
    print(recommendations[['asins', 'name', 'categories', 'prices', 'colors', 'reviews.rating', 'reviews.text']])
else:
    print(recommendations)

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall = evaluate_model(test, train, cosine_sim, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

# Define a function to plot the learning curve
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), scoring='precision')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Define a function to plot the validation curve
def plot_validation_curve(estimator, X, y, param_name, param_range, title="Validation Curve"):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range,
                                                 cv=5, scoring='precision', n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.grid()

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Example usage for plotting learning curve
from sklearn.linear_model import LogisticRegression

# We need to create target labels for the learning curve (e.g., binary labels for whether the product is recommended)
y_train = np.random.randint(0, 2, size=train.shape[0])  # Placeholder for actual binary labels
y_test = np.random.randint(0, 2, size=test.shape[0])  # Placeholder for actual binary labels

# Using a simple logistic regression for demonstration
estimator = LogisticRegression()
plot_learning_curve(estimator, tfidf_matrix_train, y_train, title="Learning Curve for Recommender System")
plt.show()

# Example usage for plotting validation curve
param_range = np.logspace(-3, 3, 7)
plot_validation_curve(estimator, tfidf_matrix_train, y_train, param_name="C", param_range=param_range, title="Validation Curve for Logistic Regression")
plt.show()

# Evaluate the model
avg_precision, avg_recall = evaluate_model(test, train, cosine_sim, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")


# In[45]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix_train, tfidf_matrix_train)

# Building the recommender function
def get_recommendations(title, train, cosine_sim, num_recommendations=10):
    if title not in train['name'].values:
        return "Product not found. Please check the name."

    idx = train[train['name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return train.iloc[top_indices]

# Define evaluation metrics
def precision_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / k

def recall_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / len(y_true)

# Evaluate the model
def evaluate_model(test_set, train_set, cosine_sim, num_recommendations=5):
    precision_scores = []
    recall_scores = []
    y_true_all = []
    y_scores_all = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        product_name = test_set.iloc[idx]['name']
        recommendations = get_recommendations(product_name, train_set, cosine_sim, num_recommendations=num_recommendations)
        if isinstance(recommendations, str):  # if the product is not found
            continue
        recommended_product_ids = recommendations['asins'].values

        y_true_all.extend([1 if true_product_id == rec_id else 0 for rec_id in recommended_product_ids])
        y_scores_all.extend([cosine_sim[idx][train_set[train_set['asins'] == rec_id].index[0]] for rec_id in recommended_product_ids])

        y_true = [true_product_id]
        y_pred = recommended_product_ids.tolist()

        precision_scores.append(precision_at_k(y_true, y_pred, num_recommendations))
        recall_scores.append(recall_at_k(y_true, y_pred, num_recommendations))

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    return avg_precision, avg_recall, y_true_all, y_scores_all

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall, y_true_all, y_scores_all = evaluate_model(test, train, cosine_sim, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true_all, y_scores_all)
pr_auc = auc(recall_vals, precision_vals)
plt.figure()
plt.plot(recall_vals, precision_vals, color='b', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Confusion Matrix
threshold = 0.5
y_pred = [1 if score >= threshold else 0 for score in y_scores_all]
conf_matrix = confusion_matrix(y_true_all, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['Negative', 'Positive'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Evaluate precision and recall at different k values
k_values = range(1, 11)
precision_scores = []
recall_scores = []

for k in k_values:
    _, _, y_true_all_k, y_scores_all_k = evaluate_model(test, train, cosine_sim, num_recommendations=k)
    precision = precision_score(y_true_all_k, [1 if score >= 0.5 else 0 for score in y_scores_all_k])
    recall = recall_score(y_true_all_k, [1 if score >= 0.5 else 0 for score in y_scores_all_k])
    precision_scores.append(precision)
    recall_scores.append(recall)

# Plot Precision@k
plt.figure(figsize=(5, 5))
plt.plot(k_values, precision_scores, marker='o', label='Precision@k')
plt.title('Precision@k for Different Values of k')
plt.xlabel('k')
plt.ylabel('Precision')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

# Plot Recall@k
plt.figure(figsize=(5, 5))
plt.plot(k_values, recall_scores, marker='o', label='Recall@k')
plt.title('Recall@k for Different Values of k')
plt.xlabel('k')
plt.ylabel('Recall')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:





# In[33]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
import math

# Assuming e_commerce DataFrame is already loaded and preprocessed

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix_train, tfidf_matrix_train)

# Building the recommender function
def get_recommendations(title, train, cosine_sim):
    if title not in train['name'].values:
        return "Product not found. Please check the name."

    idx = train[train['name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:11]]
    return train.iloc[top_indices]

# Define evaluation metrics
def precision_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / k

def recall_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / len(y_true)

def calculate_rmse(true_ratings, predicted_ratings):
    return math.sqrt(mean_squared_error(true_ratings, predicted_ratings))

def calculate_rae(true_ratings, predicted_ratings):
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    return mae / np.mean(np.abs(true_ratings - np.mean(true_ratings)))

def calculate_ndcg(true_ratings, predicted_ratings, k=5):
    return ndcg_score([true_ratings], [predicted_ratings], k=k)

# Evaluate the model
def evaluate_model(test_set, train_set, cosine_sim, num_recommendations=5):
    precision_scores = []
    recall_scores = []
    true_ratings = []
    predicted_ratings = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        product_name = test_set.iloc[idx]['name']
        true_rating = test_set.iloc[idx]['reviews.rating']
        
        recommendations = get_recommendations(product_name, train_set, cosine_sim)
        if isinstance(recommendations, str):  # if the product is not found
            continue
        
        recommended_product_ids = recommendations['asins'].values
        recommended_ratings = recommendations['reviews.rating'].values

        y_true = [true_product_id]
        y_pred = recommended_product_ids.tolist()

        precision_scores.append(precision_at_k(y_true, y_pred, num_recommendations))
        recall_scores.append(recall_at_k(y_true, y_pred, num_recommendations))
        
        true_ratings.append(true_rating)
        predicted_ratings.append(np.mean(recommended_ratings[:num_recommendations]))

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    rmse = calculate_rmse(true_ratings, predicted_ratings)
    rae = calculate_rae(true_ratings, predicted_ratings)
    ndcg = calculate_ndcg(true_ratings, predicted_ratings, k=num_recommendations)

    return avg_precision, avg_recall, rmse, rae, ndcg

# Example usage for a single product
recommendations = get_recommendations('Kindle Fire HD 7"', train, cosine_sim)
if isinstance(recommendations, pd.DataFrame):
    print(recommendations[['asins', 'name', 'categories', 'prices', 'colors', 'reviews.rating', 'reviews.text']])
else:
    print(recommendations)

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall, rmse, rae, ndcg = evaluate_model(test, train, cosine_sim, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RAE: {rae:.4f}")
print(f"NDCG@{num_recommendations}: {ndcg:.4f}")


# In[34]:


#KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Fit the Nearest Neighbor Model
nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='auto')
nearest_neighbors.fit(tfidf_matrix_train)

# Define the recommendation function
def get_recommendations(product_id, num_recommendations=5):
    if product_id not in train['asins'].values:
        return pd.DataFrame(columns=['name', 'categories', 'brand', 'reviews.text'])
    
    product_index = train[train['asins'] == product_id].index[0]
    distances, indices = nearest_neighbors.kneighbors(tfidf_matrix_train[product_index], n_neighbors=num_recommendations+1)
    recommendations = train.iloc[indices[0][1:]][['name', 'categories', 'brand', 'reviews.text']]
    return recommendations

# Define evaluation metrics
def precision_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / k

def recall_at_k(y_true, y_pred, k):
    num_relevant = len(set(y_true) & set(y_pred[:k]))
    return num_relevant / len(y_true)

# Evaluate the model
def evaluate_model(test_set, train_set, num_recommendations=5):
    precision_scores = []
    recall_scores = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        recommendations = get_recommendations(true_product_id, num_recommendations=num_recommendations)
        if recommendations.empty:
            continue
        recommended_product_ids = train_set.iloc[recommendations.index]['asins'].values

        y_true = [true_product_id]
        y_pred = recommended_product_ids.tolist()

        precision_scores.append(precision_at_k(y_true, y_pred, num_recommendations))
        recall_scores.append(recall_at_k(y_true, y_pred, num_recommendations))

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    return avg_precision, avg_recall

# Example Usage
product_id = "B00QJDU3KY"
if product_id in e_commerce['asins'].values:
    recommendations = get_recommendations(product_id, num_recommendations=5)
    print("Recommendations for Product ID", product_id)
    print(recommendations)
else:
    print(f"Product ID {product_id} not found in the dataset.")

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall = evaluate_model(test, train, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define a range of k values to evaluate
k_values = range(1, 11)

# Evaluate the model for each k
precision_scores = []
recall_scores = []

for k in k_values:
    avg_precision, avg_recall = evaluate_model(test, train, num_recommendations=k)
    precision_scores.append(avg_precision)
    recall_scores.append(avg_recall)

# Plot Precision@k
plt.figure(figsize=(10, 5))
plt.plot(k_values, precision_scores, marker='o', label='Precision@k')
plt.title('Precision@k for Different Values of k')
plt.xlabel('k')
plt.ylabel('Precision')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

# Plot Recall@k
plt.figure(figsize=(10, 5))
plt.plot(k_values, recall_scores, marker='o', label='Recall@k')
plt.title('Recall@k for Different Values of k')
plt.xlabel('k')
plt.ylabel('Recall')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()


# In[40]:


import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Assuming e_commerce dataset is already loaded

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Fit the Nearest Neighbor Model
nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='auto')
nearest_neighbors.fit(tfidf_matrix_train)

# Define the recommendation function
def get_recommendations_with_proba(product_id, num_recommendations=5):
    if product_id not in train['asins'].values:
        return pd.DataFrame(columns=['name', 'categories', 'brand', 'reviews.text']), np.array([])
    
    product_index = train[train['asins'] == product_id].index[0]
    distances, indices = nearest_neighbors.kneighbors(tfidf_matrix_train[product_index], n_neighbors=num_recommendations+1)
    recommendations = train.iloc[indices[0][1:]][['name', 'categories', 'brand', 'reviews.text']]
    return recommendations, distances[0][1:]

# Evaluate the model
def evaluate_model_with_proba(test_set, train_set, num_recommendations=5):
    y_true_all = []
    y_scores_all = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        recommendations, distances = get_recommendations_with_proba(true_product_id, num_recommendations=num_recommendations)
        if recommendations.empty:
            continue
        recommended_product_ids = train_set.iloc[recommendations.index]['asins'].values

        y_true_all.extend([1 if true_product_id == rec_id else 0 for rec_id in recommended_product_ids])
        y_scores_all.extend(1 - distances)  # Using (1 - distance) to represent similarity

    return y_true_all, y_scores_all

# Example Usage
product_id = "B00QJDU3KY"
if product_id in e_commerce['asins'].values:
    recommendations, _ = get_recommendations_with_proba(product_id, num_recommendations=5)
    print("Recommendations for Product ID", product_id)
    print(recommendations)
else:
    print(f"Product ID {product_id} not found in the dataset.")

# Evaluate the model
num_recommendations = 5
y_true_all, y_scores_all = evaluate_model_with_proba(test, train, num_recommendations)

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true_all, y_scores_all)
pr_auc = auc(recall_vals, precision_vals)
plt.figure()
plt.plot(recall_vals, precision_vals, color='b', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Fit the Nearest Neighbor Model
nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='auto')
nearest_neighbors.fit(tfidf_matrix_train)

# Define the recommendation function
def get_recommendations(product_id, num_recommendations=5):
    if product_id not in train['asins'].values:
        return pd.DataFrame(columns=['name', 'categories', 'brand', 'reviews.text'])
    
    product_index = train[train['asins'] == product_id].index[0]
    distances, indices = nearest_neighbors.kneighbors(tfidf_matrix_train[product_index], n_neighbors=num_recommendations+1)
    recommendations = train.iloc[indices[0][1:]][['name', 'categories', 'brand', 'reviews.text']]
    return recommendations

# Generate predictions and true labels
def generate_predictions(test_set, train_set, num_recommendations=5):
    y_true = []
    y_pred = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        recommendations = get_recommendations(true_product_id, num_recommendations=num_recommendations)
        if recommendations.empty:
            continue
        recommended_product_ids = train_set.iloc[recommendations.index]['asins'].values

        y_true.append(true_product_id)
        y_pred.append(recommended_product_ids.tolist())

    return y_true, y_pred

# Flatten the lists and create binary labels for confusion matrix
def create_binary_labels(y_true, y_pred, num_recommendations):
    y_true_binary = []
    y_pred_binary = []

    for i in range(len(y_true)):
        true_product = y_true[i]
        recommended_products = y_pred[i]

        for j in range(num_recommendations):
            if j < len(recommended_products):
                y_pred_binary.append(1 if recommended_products[j] == true_product else 0)
                y_true_binary.append(1 if recommended_products[j] == true_product else 0)
            else:
                y_pred_binary.append(0)
                y_true_binary.append(0)

    return y_true_binary, y_pred_binary

# Generate predictions
num_recommendations = 5
y_true, y_pred = generate_predictions(test, train, num_recommendations)

# Create binary labels
y_true_binary, y_pred_binary = create_binary_labels(y_true, y_pred, num_recommendations)

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_true_binary, y_pred_binary)
plt.figure(figsize=(6, 6))  # Adjust the figure size here
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['Negative', 'Positive'])

# Print numbers in the confusion matrix cells
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Evaluate the model
avg_precision, avg_recall = evaluate_model(test, train, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")


# In[36]:


#Decision tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc# Convert ratings to a binary outcome
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import re
from sklearn.pipeline import Pipeline
e_commerce['final'] = e_commerce['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(e_commerce['combined_features'], e_commerce['final'], test_size=0.2, random_state=42)

# Creating a pipeline with a TF-IDF vectorizer and a decision tree classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])

# Training the model
pipeline.fit(X_train, y_train)

# Evaluating the model
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision of the Decision Tree model: {precision:.4f}')
print(f'Recall of the Decision Tree model: {recall:.4f}')
print(f'F1 Score of the Decision Tree model: {f1:.4f}')

# Plotting Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_vals, precision_vals)
plt.figure()
plt.plot(recall_vals, precision_vals, color='b', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()


# In[37]:


from sklearn.metrics import confusion_matrix, roc_curve, auc

# Evaluate the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision of the Decision Tree model: {precision:.4f}')
print(f'Recall of the Decision Tree model: {recall:.4f}')
print(f'F1 Score of the Decision Tree model: {f1:.4f}')

# Plotting Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_vals, precision_vals)
plt.figure()
plt.plot(recall_vals, precision_vals, color='b', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Plotting ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['Negative', 'Positive'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, roc_curve
# 
# # Assuming e_commerce dataset is already loaded and preprocessed
# 
# # Convert ratings to a binary outcome
# e_commerce['final'] = e_commerce['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)
# 
# # Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(e_commerce['combined_features'], e_commerce['final'], test_size=0.2, random_state=42)
# 
# # Creating a pipeline with a TF-IDF vectorizer and a decision tree classifier
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words='english')),
#     ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
# ])
# 
# # Training the model
# pipeline.fit(X_train, y_train)
# 
# # Getting predicted probabilities
# y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
# 
# # Define a range of k values to evaluate
# k_values = range(1, 11)
# 
# # Initialize lists to store precision and recall values
# precision_scores = []
# recall_scores = []
# 
# for k in k_values:
#     # Binarize the predicted probabilities with different thresholds
#     y_pred_k = (y_pred_proba >= np.sort(y_pred_proba)[-k]).astype(int)
#     
#     precision = precision_score(y_test, y_pred_k)
#     recall = recall_score(y_test, y_pred_k)
#     
#     precision_scores.append(precision)
#     recall_scores.append(recall)
# 
#     print(f'k={k}: Precision = {precision:.4f}, Recall = {recall:.4f}')
# 
# # Plot Precision@k
# plt.figure(figsize=(10, 5))
# plt.plot(k_values, precision_scores, marker='o', label='Precision@k')
# plt.title('Precision@k for Different Values of k')
# plt.xlabel('k')
# plt.ylabel('Precision')
# plt.xticks(k_values)
# plt.grid(True)
# plt.legend()
# plt.show()
# 
# # Plot Recall@k
# plt.figure(figsize=(10, 5))
# plt.plot(k_values, recall_scores, marker='o', label='Recall@k')
# plt.title('Recall@k for Different Values of k')
# plt.xlabel('k')
# plt.ylabel('Recall')
# plt.xticks(k_values)
# plt.grid(True)
# plt.legend()
# plt.show()
# 

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, roc_curve

# Assuming e_commerce dataset is already loaded and preprocessed

# Convert ratings to a binary outcome
e_commerce['final'] = e_commerce['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(e_commerce['combined_features'], e_commerce['final'], test_size=0.2, random_state=42)

# Creating a pipeline with a TF-IDF vectorizer and a decision tree classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])

# Training the model
pipeline.fit(X_train, y_train)

# Getting predicted probabilities
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Define a range of k values to evaluate
k_values = range(1, 11)

# Initialize lists to store precision and recall values
precision_scores = []
recall_scores = []

for k in k_values:
    # Binarize the predicted probabilities with different thresholds
    y_pred_k = (y_pred_proba >= np.sort(y_pred_proba)[-k]).astype(int)
    
    precision = precision_score(y_test, y_pred_k)
    recall = recall_score(y_test, y_pred_k)
    
    precision_scores.append(precision)
    recall_scores.append(recall)

    print(f'k={k}: Precision = {precision:.4f}, Recall = {recall:.4f}')

# Plot Precision@k
plt.figure(figsize=(10, 5))
plt.plot(k_values, precision_scores, marker='o', label='Precision@k')
plt.title('Precision@k for Different Values of k')
plt.xlabel('k')
plt.ylabel('Precision')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

# Plot Recall@k
plt.figure(figsize=(10, 5))
plt.plot(k_values, recall_scores, marker='o', label='Recall@k')
plt.title('Recall@k for Different Values of k')
plt.xlabel('k')
plt.ylabel('Recall')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




