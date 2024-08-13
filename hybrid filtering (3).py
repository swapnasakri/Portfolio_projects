#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


e_commerce=pd.read_csv('7817_1.csv')


# In[3]:


#function to parse the prices from JSon string
def parse_price(prices):
    try:
        prices=json.loads(prices.replace("'" ,'"'))
        return min(price['amountMin'] for p in prices if 'amountMin' in p)
    except:
        return np.nan
    
    


# In[4]:


#parse the date
def parse_date(date):
    try:
        return datetime.strptime(date,'%Y-%m-%dT%H:%M:%SZ')
    except:
        return np.nan
        


# In[5]:


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


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_ma = tfidf.fit_transform(e_commerce['combined_features'])


# In[7]:


from sklearn.metrics.pairwise import cosine_similarity
# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_ma, tfidf_ma)


# In[8]:


import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

ranking = e_commerce[['reviews.username', 'id', 'reviews.rating']].dropna()

# Remove rows with non-numeric or missing ratings
ranking = ranking[pd.to_numeric(ranking['reviews.rating'], errors='coerce').notnull()]

# Convert ratings to float
ranking['reviews.rating'] = ranking['reviews.rating'].astype(float)

# Rename columns for Surprise
ranking.columns = ['userId', 'itemId', 'rating']

# Load data into Surprise dataset
reader = Reader(rating_scale=(1, 5))
surprise = Dataset.load_from_df(ranking, reader)

# Split data into training and testing sets
trainset, testset = train_test_split(surprise, test_size=0.2)

# Train SVD model
model_cf = SVD()
model_cf.fit(trainset)


# In[9]:


import numpy as np

# Function to get CF predictions
def c_pred(user_id, item_id, model
          ):
    try:
        pred = model.predict(user_id, item_id)
        return pred.est
    except:
        return 0


# In[10]:


# Function to get CB predictions
def collaborative_pred(item_id, cosine_sim_matrix, product_info, n=10):
    try:
        dx = product_info.index[product_info['itemId'] == item_id].tolist()[0]
        sim_s = list(enumerate(cosine_sim_matrix[dx]))
        sim_s = sorted(sim_s, key=lambda x: x[1], reverse=True)
        sim_s = sim_s[1:n+1]
        product_indice = [i[0] for i in sim_s]
        return product_info.iloc[product_indice]
    except IndexError:
        return pd.DataFrame()


# In[11]:


# Hybrid Recommender System
def hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, product_info, alpha=0.5):
    cf_s = c_pred(user_id, item_id, model_cf)
    cb_predictions = collaborative_pred(item_id, cosine_sim_matrix, product_info)
    if not cb_predictions.empty:
        cb_sc = cb_predictions['rating'].mean()
    else:
        cb_s = 0
    
    hybrid_s = alpha * cf_s + (1 - alpha) * cb_s
    return hybrid_s


# In[12]:


# Define items_data for cb_predict function
# Define items_data for cb_predict function
product_info = e_commerce[['asins', 'reviews.rating']].drop_duplicates(subset='asins')
product_info.columns = ['itemId', 'rating']


# In[13]:


# Testing the hybrid recommender system
test_user_id = 'Cristina M'  # Example user from the dataset
test_item_id = 'B00QJDU3KY'  # Example item from the dataset

hybrid_s = hybrid_recommend(test_user_id, test_item_id, model_cf, cosine_sim_matrix, product_info, alpha=0.7)
print(f'Hybrid Recommendation Score for user {test_user_id} and item {test_item_id}: {hybrid_s}')


# In[14]:


# Ensure the test_item_id exists in items_data
if test_item_id in product_info['itemId'].values:
    hybrid_s = hybrid_recommend(test_user_id, test_item_id, model_cf, cosine_sim_matrix, product_info, alpha=0.7)
    print(f'Hybrid Recommendation Score for user {test_user_id} and item {test_item_id}: {hybrid_s}')
else:
    print(f'Item ID {test_item_id} not found in the dataset.')


# In[15]:


# Function to get top N recommendations for a user
def get_top_recommendations(user_id, n=5):
    user_ranking = ranking[ranking['userId'] == user_id]
    unranked_items = product_info[~product_info['itemId'].isin(user_ranking['itemId'])]
    
    recommendations = []
    for item_id in unranked_items['itemId']:
        mark = hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, product_info)
        recommendations.append((item_id, mark))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


# In[16]:


# Example usage
test_user_id = 'Cristina M'
top_recommendations = get_top_recommendations(test_user_id)

print(f"Top 5 recommendations for user {test_user_id}:")
for item_id, mark in top_recommendations:
    item_name = e_commerce[e_commerce['asins'] == item_id]['name'].iloc[0]
    print(f"{item_name} (ID: {item_id}): Score = {mark:.4f}")


# In[17]:


def cb_recommend(item_id, cosine_sim_matrix, product_info, n=5):
    try:
        dx = product_info.index[product_info['asins'] == item_id].tolist()[0]
        sim_s = list(enumerate(cosine_sim_matrix[idx]))
        sim_s = sorted(sim_s, key=lambda x: x[1], reverse=True)
        sim_s = sim_s[1:]  
        product_indice = []
        recommended_items = []
        for i, mark in sim_s:
            if product_info.iloc[i]['asins'] not in recommended_items:
                product_indice.append(i)
                recommended_items.append(product_info.iloc[i]['asins'])
                if len(recommended_items) == n:
                    break
        return [(product_info.iloc[i]['asins'], sim_s[idx][1]) for dx, i in enumerate(product_indice)]
    except IndexError:
        return []


# In[18]:


from sklearn.metrics import mean_absolute_error, ndcg_score
import numpy as np

# Calculate MAE
def calculate_mae(true_ranking, predicted_ranking):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return mean_absolute_error(true_ranking, predicted_ranking)

# Calculate NDCG
def calculate_ndcg(true_ranking, predicted_ranking, k=10):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return ndcg_score([true_ranking], [predicted_ranking], k=k)

# Pure Collaborative Filtering
def cf_recommend(user_id, model, product_to_forecast, n=5):
    forecasts = [model.predict(user_id, item_id) for item_id in product_to_forecast]
    n_t = sorted(forecasts, key=lambda x: x.est, reverse=True)[:n]
    return [(pred.iid, pred.est) for pred in n_t]

# Pure Content-Based Filtering
def cb_recommend(item_id, cosine_sim_matrix, product_info, n=5):
    try:
        idx = product_info.index[product_info['asins'] == item_id].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        product_indices = [i[0] for i in sim_scores]
        return [(product_info.iloc[i]['asins'], sim_scores[dx][1]) for dx, i in enumerate(product_indices) if i < len(product_info)]
    except IndexError:
        return []

# Hybrid Recommender
def hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, product_info, alpha=0.5):
    cf_pred = model_cf.predict(user_id, item_id).est
    cb_preds = cb_recommend(item_id, cosine_sim_matrix, product_info, n=1)
    cb_pred = cb_preds[0][1] if cb_preds else cf_pred
    return alpha * cf_pred + (1 - alpha) * cb_pred

# Perform A/B testing and evaluation
def ab_test_and_evaluate(testset, model_cf, cosine_sim_matrix, product_info):
    cf_pred = []
    cb_pred = []
    hybrid_pred = []
    true_ratings = []

    for r in testset:
        ud, idi, true_rating = r  # Assuming the tuple is in this order

        try:
            # Collaborative Filtering
            cf_p = model_cf.predict(ud, idi).est
            cf_pred.append(cf_p)

            # Content-Based Filtering
            cb_preds = cb_recommend(idi, cosine_sim_matrix, product_info, n=1)
            cb_p = cb_preds[0][1] if cb_preds else cf_p
            cb_pred.append(cb_p)

            # Hybrid
            hybrid_p = hybrid_recommend(ud, idi, model_cf, cosine_sim_matrix, product_info)
            hybrid_pred.append(hybrid_p)

            true_ratings.append(true_rating)
        except Exception as e:
            print(f"Error processing user {ud}, item {idi}: {str(e)}")

    # Calculate metrics
    cf_mae = calculate_mae(true_ratings, cf_pred)
    cb_mae = calculate_mae(true_ratings, cb_pred)
    hybrid_mae = calculate_mae(true_ratings, hybrid_pred)

    cf_ndcg = calculate_ndcg(true_ratings, cf_pred)
    cb_ndcg = calculate_ndcg(true_ratings, cb_pred)
    hybrid_ndcg = calculate_ndcg(true_ratings, hybrid_pred)

    return {
        'CF': {'MAE': cf_mae, 'NDCG': cf_ndcg},
        'CB': {'MAE': cb_mae, 'NDCG': cb_ndcg},
        'Hybrid': {'MAE': hybrid_mae, 'NDCG': hybrid_ndcg}
    }

# Main execution
if __name__ == "__main__":
    # Perform A/B testing and evaluation
    results = ab_test_and_evaluate(testset, model_cf, cosine_sim_matrix, e_commerce)

    # Print results
    for m, metrics in results.items():
        print(f"{m} Results:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  NDCG: {metrics['NDCG']:.4f}")
        print()

    # Example recommendation
    test_user, test_item, _ = testset[0]  # Assuming the first element of testset

    print(f"Recommendations for user {test_user}:")
    cf_recs = cf_recommend(test_user, model_cf, e_commerce['asins'].unique(), n=5)
    cb_recs = cb_recommend(test_item, cosine_sim_matrix, e_commerce, n=5)
    hybrid_recs = [hybrid_recommend(test_user, item, model_cf, cosine_sim_matrix, e_commerce) for item, _ in cf_recs]

    print("CF Recommendations:")
    for item, score in cf_recs:
        print(f"  Item: {item}, Score: {score:.4f}")

    print("\nCB Recommendations:")
    for item, score in cb_recs:
        print(f"  Item: {item}, Score: {score:.4f}")

    print("\nHybrid Recommendations:")
    for item, score in zip([item for item, _ in cf_recs], hybrid_recs):
        print(f"  Item: {item}, Score: {score:.4f}")


# In[19]:


from sklearn.metrics import mean_absolute_error, ndcg_score
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, SVD

# Calculate MAE
def calculate_mae(true_ranking, predicted_ranking):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return mean_absolute_error(true_ranking, predicted_ranking)

# Calculate NDCG
def calculate_ndcg(true_ranking, predicted_ranking, k=10):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return ndcg_score([true_ranking], [predicted_ranking], k=k)

# Pure Collaborative Filtering
def cf_recommend(user_id, model, product_to_forecast, n=5):
    forecasts = [model.predict(user_id, item_id) for item_id in product_to_forecast]
    n_t = sorted(forecasts, key=lambda x: x.est, reverse=True)[:n]
    return [(pred.iid, pred.est) for pred in n_t]

# Pure Content-Based Filtering
def cb_recommend(item_id, cosine_sim_matrix, product_info, n=5):
    try:
        idx = product_info.index[product_info['asins'] == item_id].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]  
        product_indices = []
        recommended_items = []
        for i, mark in sim_scores:
            if product_info.iloc[i]['asins'] not in recommended_items:
                product_indices.append(i)
                recommended_items.append(product_info.iloc[i]['asins'])
                if len(recommended_items) == n:
                    break
        return [(product_info.iloc[i]['asins'], sim_scores[dx][1]) for dx, i in enumerate(product_indices)]
    except IndexError:
        return []

# Hybrid Recommender
def hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, product_info, alpha=0.5):
    cf_pred = model_cf.predict(user_id, item_id).est
    cb_preds = cb_recommend(item_id, cosine_sim_matrix, product_info, n=1)
    cb_pred = cb_preds[0][1] if cb_preds else cf_pred
    return alpha * cf_pred + (1 - alpha) * cb_pred

# Perform A/B testing and evaluation
def ab_test_and_evaluate(testset, model_cf, cosine_sim_matrix, product_info):
    cf_pred = []
    cb_pred = []
    hybrid_pred = []
    true_ratings = []

    for r in testset:
        ud, idi, true_rating = r  # Assuming the tuple is in this order

        try:
            # Collaborative Filtering
            cf_p = model_cf.predict(ud, idi).est
            cf_pred.append(cf_p)

            # Content-Based Filtering
            cb_preds = cb_recommend(idi, cosine_sim_matrix, product_info, n=1)
            cb_p = cb_preds[0][1] if cb_preds else cf_p
            cb_pred.append(cb_p)

            # Hybrid
            hybrid_p = hybrid_recommend(ud, idi, model_cf, cosine_sim_matrix, product_info)
            hybrid_pred.append(hybrid_p)

            true_ratings.append(true_rating)
        except Exception as e:
            print(f"Error processing user {ud}, item {idi}: {str(e)}")

    # Calculate metrics
    cf_mae = calculate_mae(true_ratings, cf_pred)
    cb_mae = calculate_mae(true_ratings, cb_pred)
    hybrid_mae = calculate_mae(true_ratings, hybrid_pred)

    cf_ndcg = calculate_ndcg(true_ratings, cf_pred)
    cb_ndcg = calculate_ndcg(true_ratings, cb_pred)
    hybrid_ndcg = calculate_ndcg(true_ratings, hybrid_pred)

    return {
        'CF': {'MAE': cf_mae, 'NDCG': cf_ndcg},
        'CB': {'MAE': cb_mae, 'NDCG': cb_ndcg},
        'Hybrid': {'MAE': hybrid_mae, 'NDCG': hybrid_ndcg}
    }

# Main execution
if __name__ == "__main__":
    # Assuming 'testset', 'model_cf', 'cosine_sim_matrix', and 'e_commerce' are already defined
    # Perform A/B testing and evaluation
    results = ab_test_and_evaluate(testset, model_cf, cosine_sim_matrix, e_commerce)

    # Print results
    for m, metrics in results.items():
        print(f"{m} Results:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  NDCG: {metrics['NDCG']:.4f}")
        print()

    # Example recommendation
    test_user, test_item, _ = testset[0]  # Assuming the first element of testset

    print(f"Recommendations for user {test_user}:")
    cf_recs = cf_recommend(test_user, model_cf, e_commerce['asins'].unique(), n=5)
    cb_recs = cb_recommend(test_item, cosine_sim_matrix, e_commerce, n=5)
    hybrid_recs = [(item, hybrid_recommend(test_user, item, model_cf, cosine_sim_matrix, e_commerce)) for item, _ in cf_recs]

    print("CF Recommendations:")
    for item, score in cf_recs:
        print(f"  Item: {item}, Score: {score:.4f}")

    print("\nCB Recommendations:")
    for item, score in cb_recs:
        print(f"  Item: {item}, Score: {score:.4f}")

    print("\nHybrid Recommendations:")
    for item, score in hybrid_recs:
        print(f"  Item: {item}, Score: {score:.4f}")


# In[20]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, ndcg_score
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, SVD

# Calculate MAE
def calculate_mae(true_ranking, predicted_ranking):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return mean_absolute_error(true_ranking, predicted_ranking)

# Calculate NDCG
def calculate_ndcg(true_ranking, predicted_ranking, k=10):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return ndcg_score([true_ranking], [predicted_ranking], k=k)

# Pure Collaborative Filtering
def cf_recommend(user_id, model, product_to_forecast, n=5):
    forecasts = [model.predict(user_id, item_id) for item_id in product_to_forecast]
    n_t = sorted(forecasts, key=lambda x: x.est, reverse=True)[:n]
    return [(pred.iid, pred.est) for pred in n_t]

# Pure Content-Based Filtering
def cb_recommend(item_id, cosine_sim_matrix, product_info, n=5):
    try:
        idx = product_info.index[product_info['asins'] == item_id].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]  
        product_indices = []
        recommended_items = []
        for i, mark in sim_scores:
            if product_info.iloc[i]['asins'] not in recommended_items:
                product_indices.append(i)
                recommended_items.append(product_info.iloc[i]['asins'])
                if len(recommended_items) == n:
                    break
        return [(product_info.iloc[i]['asins'], sim_scores[dx][1]) for dx, i in enumerate(product_indices)]
    except IndexError:
        return []

# Hybrid Recommender
def hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, product_info, alpha=0.5):
    cf_pred = model_cf.predict(user_id, item_id).est
    cb_preds = cb_recommend(item_id, cosine_sim_matrix, product_info, n=1)
    cb_pred = cb_preds[0][1] if cb_preds else cf_pred
    return alpha * cf_pred + (1 - alpha) * cb_pred

# Perform A/B testing and evaluation
def ab_test_and_evaluate(testset, model_cf, cosine_sim_matrix, product_info):
    cf_pred = []
    cb_pred = []
    hybrid_pred = []
    true_ratings = []

    for r in testset:
        ud, idi, true_rating = r  # Assuming the tuple is in this order

        try:
            # Collaborative Filtering
            cf_p = model_cf.predict(ud, idi).est
            cf_pred.append(cf_p)

            # Content-Based Filtering
            cb_preds = cb_recommend(idi, cosine_sim_matrix, product_info, n=1)
            cb_p = cb_preds[0][1] if cb_preds else cf_p
            cb_pred.append(cb_p)

            # Hybrid
            hybrid_p = hybrid_recommend(ud, idi, model_cf, cosine_sim_matrix, product_info)
            hybrid_pred.append(hybrid_p)

            true_ratings.append(true_rating)
        except Exception as e:
            print(f"Error processing user {ud}, item {idi}: {str(e)}")

    # Calculate metrics
    cf_mae = calculate_mae(true_ratings, cf_pred)
    cb_mae = calculate_mae(true_ratings, cb_pred)
    hybrid_mae = calculate_mae(true_ratings, hybrid_pred)

    cf_ndcg = calculate_ndcg(true_ratings, cf_pred)
    cb_ndcg = calculate_ndcg(true_ratings, cb_pred)
    hybrid_ndcg = calculate_ndcg(true_ratings, hybrid_pred)

    return {
        'CF': {'MAE': cf_mae, 'NDCG': cf_ndcg},
        'CB': {'MAE': cb_mae, 'NDCG': cb_ndcg},
        'Hybrid': {'MAE': hybrid_mae, 'NDCG': hybrid_ndcg}
    }

# Plotting function
def plot_results(results):
    metrics = ['MAE', 'NDCG']
    models = list(results.keys())
    
    for metric in metrics:
        values = [results[model][metric] for model in models]
        
        plt.figure(figsize=(7, 4))
        plt.bar(models, values, color=['skyblue', 'lightgreen', 'coral'])
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Assuming 'testset', 'model_cf', 'cosine_sim_matrix', and 'e_commerce' are already defined
    # Perform A/B testing and evaluation
    results = ab_test_and_evaluate(testset, model_cf, cosine_sim_matrix, e_commerce)

    # Print results
    for m, metrics in results.items():
        print(f"{m} Results:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  NDCG: {metrics['NDCG']:.4f}")
        print()

    # Plot results
    plot_results(results)

    # Example recommendation
    test_user, test_item, _ = testset[0]  # Assuming the first element of testset

    print(f"Recommendations for user {test_user}:")
    cf_recs = cf_recommend(test_user, model_cf, e_commerce['asins'].unique(), n=5)
    cb_recs = cb_recommend(test_item, cosine_sim_matrix, e_commerce, n=5)
    hybrid_recs = [(item, hybrid_recommend(test_user, item, model_cf, cosine_sim_matrix, e_commerce)) for item, _ in cf_recs]

    print("CF Recommendations:")
    for item, score in cf_recs:
        print(f"  Item: {item}, Score: {score:.4f}")

    print("\nCB Recommendations:")
    for item, score in cb_recs:
        print(f"  Item: {item}, Score: {score:.4f}")

    print("\nHybrid Recommendations:")
    for item, score in hybrid_recs:
        print(f"  Item: {item}, Score: {score:.4f}")


# In[21]:


def content_based_recommendations_for_new_user(item_data, cosine_sim_matrix, n=5):
    # For a new user, we'll recommend the most popular items based on average ratings
    popular_items = item_data.sort_values('avg_rating', ascending=False).head(n)
    
    recommendations = []
    for _, item in popular_items.iterrows():
        similar_items = cb_recommend(item['asins'], cosine_sim_matrix, item_data, n=1)
        if similar_items:
            recommendations.append((item['asins'], similar_items[0][1]))
    
    return recommendations[:n]


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def metadata_based_recommendations_for_new_item(new_item, item_data, n=5):
    # Combine relevant metadata fields
    item_data['metadata'] = item_data['name'] + ' ' + item_data['categories'] + ' ' + item_data['brand']
    new_item_metadata = new_item['name'] + ' ' + new_item['categories'] + ' ' + new_item['brand']
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(item_data['metadata'].tolist() + [new_item_metadata])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Get top similar items
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:n]
    
    # Get item indices
    item_indices = [i[0] for i in sim_scores]
    
    # Return recommended items
    return [(item_data.iloc[i]['asins'], sim_scores[idx][1]) for idx, i in enumerate(item_indices)]


# In[23]:


def hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, items_data, n=5, alpha=0.5):
    cf_preds = cf_recommend(user_id, model_cf, items_data['asins'].unique(), n)
    cb_preds = cb_recommend(item_id, cosine_sim_matrix, items_data, n)
    
    # Combine CF and CB predictions
    hybrid_preds = {}
    for item, score in cf_preds:
        hybrid_preds[item] = alpha * score
    for item, score in cb_preds:
        if item in hybrid_preds:
            hybrid_preds[item] += (1 - alpha) * score
        else:
            hybrid_preds[item] = (1 - alpha) * score
    
    # Sort and return top n recommendations
    sorted_preds = sorted(hybrid_preds.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds[:n]



# In[24]:


def get_recommendations(user_id, item_id, model_cf, cosine_sim_matrix, item_data, n=5):
    if user_id not in model_cf.trainset.all_users():
        # New user
        return content_based_recommendations_for_new_user(item_data, cosine_sim_matrix, n)
    elif item_id not in item_data['asins'].unique():
        # New item
        print(f"Warning: Item {item_id} not found in the dataset. Returning general recommendations.")
        return general_recommendations(item_data, n)
    else:
        # Existing user and item
        cf_pred = model_cf.predict(user_id, item_id).est
        cb_preds = content_based_recommendations(item_id, cosine_sim_matrix, item_data, n)
        
        # Combine CF and CB predictions
        hybrid_preds = []
        for cb_item, cb_score in cb_preds:
            cf_score = model_cf.predict(user_id, cb_item).est
            hybrid_score = (cf_score + cb_score) / 2
            hybrid_preds.append((cb_item, hybrid_score))
        
        return sorted(hybrid_preds, key=lambda x: x[1], reverse=True)[:n]

def general_recommendations(item_data, n=5):
    # This function returns the top N most popular items
    top_items = item_data.groupby('asins')['reviews.rating'].mean().sort_values(ascending=False).head(n)
    return [(item, score) for item, score in top_items.items()]

def content_based_recommendations_for_new_user(item_data, cosine_sim_matrix, n=5):
    # This function returns the top N most popular items
    return general_recommendations(item_data, n)

def content_based_recommendations(item_id, cosine_sim_matrix, item_data, n=5):
    idx = item_data.index[item_data['asins'] == item_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    item_indices = [i[0] for i in sim_scores]
    return [(item_data.iloc[i]['asins'], sim_scores[idx][1]) for idx, i in enumerate(item_indices)]


# In[25]:


if __name__ == "__main__":
    # Example usage for existing user and item
    test_user, test_item, _ = testset[0]  # Assuming testset is a list of tuples (user, item, rating)
    print(f"\nRecommendations for existing user {test_user} and item {test_item}:")
    recommendations = get_recommendations(test_user, test_item, model_cf, cosine_sim_matrix, e_commerce)
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")

    # Example usage for new user
    new_user = "new_user_123"
    print(f"\nRecommendations for new user {new_user}:")
    recommendations = get_recommendations(new_user, test_item, model_cf, cosine_sim_matrix, e_commerce)
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")

    # Example usage for new item
    new_item = 'NEW_ITEM_001'
    print(f"\nRecommendations for new item {new_item}:")
    recommendations = get_recommendations(test_user, new_item, model_cf, cosine_sim_matrix, e_commerce)
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")


# In[26]:


from sklearn.metrics import mean_absolute_error, ndcg_score
import numpy as np
import matplotlib.pyplot as plt

# Calculate MAE
def calculate_mae(true_ranking, predicted_ranking):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return mean_absolute_error(true_ranking, predicted_ranking)

# Calculate NDCG
def calculate_ndcg(true_ranking, predicted_ranking, k=10):
    true_ranking = np.array(true_ranking)  # Ensure array-like input
    predicted_ranking = np.array(predicted_ranking)  # Ensure array-like input
    return ndcg_score([true_ranking], [predicted_ranking], k=k)

# Pure Collaborative Filtering
def cf_recommend(user_id, model, product_to_forecast, n=5):
    forecasts = [model.predict(user_id, item_id) for item_id in product_to_forecast]
    n_t = sorted(forecasts, key=lambda x: x.est, reverse=True)[:n]
    return [(pred.iid, pred.est) for pred in n_t]

# Pure Content-Based Filtering
def cb_recommend(item_id, cosine_sim_matrix, product_info, n=5):
    try:
        idx = product_info.index[product_info['asins'] == item_id].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        product_indices = [i[0] for i in sim_scores]
        return [(product_info.iloc[i]['asins'], sim_scores[dx][1]) for dx, i in enumerate(product_indices) if i < len(product_info)]
    except IndexError:
        return []

# Hybrid Recommender
def hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, items_data, n=5, alpha=0.5):
    cf_preds = cf_recommend(user_id, model_cf, items_data['asins'].unique(), n)
    cb_preds = cb_recommend(item_id, cosine_sim_matrix, items_data, n)
    
    # Combine CF and CB predictions
    hybrid_preds = {}
    for item, score in cf_preds:
        hybrid_preds[item] = alpha * score
    for item, score in cb_preds:
        if item in hybrid_preds:
            hybrid_preds[item] += (1 - alpha) * score
        else:
            hybrid_preds[item] = (1 - alpha) * score
    
    # Sort and return top n recommendations
    sorted_preds = sorted(hybrid_preds.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds[:n]

def get_recommendations(user_id, item_id, model_cf, cosine_sim_matrix, item_data, n=5):
    if user_id not in model_cf.trainset.all_users():
        # New user
        return content_based_recommendations_for_new_user(item_data, cosine_sim_matrix, n)
    elif item_id not in item_data['asins'].unique():
        # New item
        print(f"Warning: Item {item_id} not found in the dataset. Returning general recommendations.")
        return general_recommendations(item_data, n)
    else:
        # Existing user and item
        cf_pred = model_cf.predict(user_id, item_id).est
        cb_preds = content_based_recommendations(item_id, cosine_sim_matrix, item_data, n)
        
        # Combine CF and CB predictions
        hybrid_preds = []
        for cb_item, cb_score in cb_preds:
            cf_score = model_cf.predict(user_id, cb_item).est
            hybrid_score = (cf_score + cb_score) / 2
            hybrid_preds.append((cb_item, hybrid_score))
        
        return sorted(hybrid_preds, key=lambda x: x[1], reverse=True)[:n]

def general_recommendations(item_data, n=5):
    # This function returns the top N most popular items
    top_items = item_data.groupby('asins')['reviews.rating'].mean().sort_values(ascending=False).head(n)
    return [(item, score) for item, score in top_items.items()]

def content_based_recommendations_for_new_user(item_data, cosine_sim_matrix, n=5):
    # This function returns the top N most popular items
    return general_recommendations(item_data, n)

def content_based_recommendations(item_id, cosine_sim_matrix, item_data, n=5):
    idx = item_data.index[item_data['asins'] == item_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    item_indices = [i[0] for i in sim_scores]
    return [(item_data.iloc[i]['asins'], sim_scores[idx][1]) for idx, i in enumerate(item_indices)]

def evaluate_recommendations(testset, model_cf, cosine_sim_matrix, item_data):
    cf_mae, cb_mae, hybrid_mae = [], [], []
    cf_ndcg, cb_ndcg, hybrid_ndcg = [], [], []

    for user, item, true_rating in testset:
        cf_preds = get_recommendations(user, item, model_cf, cosine_sim_matrix, item_data, n=5)
        cb_preds = cb_recommend(item, cosine_sim_matrix, item_data, n=5)
        hybrid_preds = hybrid_recommend(user, item, model_cf, cosine_sim_matrix, item_data, n=5)

        cf_ratings = [pred[1] for pred in cf_preds]
        cb_ratings = [pred[1] for pred in cb_preds]
        hybrid_ratings = [pred[1] for pred in hybrid_preds]

        # Ensure that there are ratings to evaluate
        if cf_ratings:
            cf_mae.append(calculate_mae([true_rating]*len(cf_ratings), cf_ratings))
            cf_ndcg.append(calculate_ndcg([true_rating]*len(cf_ratings), cf_ratings))

        if cb_ratings:
            cb_mae.append(calculate_mae([true_rating]*len(cb_ratings), cb_ratings))
            cb_ndcg.append(calculate_ndcg([true_rating]*len(cb_ratings), cb_ratings))

        if hybrid_ratings:
            hybrid_mae.append(calculate_mae([true_rating]*len(hybrid_ratings), hybrid_ratings))
            hybrid_ndcg.append(calculate_ndcg([true_rating]*len(hybrid_ratings), hybrid_ratings))

    return {
        'CF': {'MAE': np.mean(cf_mae) if cf_mae else float('nan'), 'NDCG': np.mean(cf_ndcg) if cf_ndcg else float('nan')},
        'CB': {'MAE': np.mean(cb_mae) if cb_mae else float('nan'), 'NDCG': np.mean(cb_ndcg) if cb_ndcg else float('nan')},
        'Hybrid': {'MAE': np.mean(hybrid_mae) if hybrid_mae else float('nan'), 'NDCG': np.mean(hybrid_ndcg) if hybrid_ndcg else float('nan')}
    }

if __name__ == "__main__":
    # Example usage for existing user and item
    test_user, test_item, _ = testset[0]  # Assuming testset is a list of tuples (user, item, rating)
    print(f"\nRecommendations for existing user {test_user} and item {test_item}:")
    recommendations = get_recommendations(test_user, test_item, model_cf, cosine_sim_matrix, e_commerce)
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")

    # Example usage for new user
    new_user = "new_user_123"
    print(f"\nRecommendations for new user {new_user}:")
    recommendations = get_recommendations(new_user, test_item, model_cf, cosine_sim_matrix, e_commerce)
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")

    # Example usage for new item
    new_item = 'NEW_ITEM_001'
    print(f"\nRecommendations for new item {new_item}:")
    recommendations = get_recommendations(test_user, new_item, model_cf, cosine_sim_matrix, e_commerce)
    for item, score in recommendations:
        print(f"  Item: {item}, Score: {score:.4f}")

    # Evaluate the model
    evaluation_results = evaluate_recommendations(testset, model_cf, cosine_sim_matrix, e_commerce)

    print("\nEvaluation Results:")
    for model_type, metrics in evaluation_results.items():
        print(f"{model_type} - MAE: {metrics['MAE']:.4f}, NDCG: {metrics['NDCG']:.4f}")

    # Plot the evaluation results
    models = ['CF', 'CB', 'Hybrid']
    mae_scores = [evaluation_results[model]['MAE'] for model in models]
    ndcg_scores = [evaluation_results[model]['NDCG'] for model in models]

    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.bar(models, mae_scores, color=['skyblue', 'lightgreen', 'salmon'])
    plt.xlabel('Models')
    plt.ylabel('MAE')
    plt.title('MAE Comparison')

    plt.subplot(1, 2, 2)
    plt.bar(models, ndcg_scores, color=['skyblue', 'lightgreen', 'salmon'])
    plt.xlabel('Models')
    plt.ylabel('NDCG')
    plt.title('NDCG Comparison')

    plt.tight_layout()
    plt.show()


# In[27]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, GridSearchCV as SurpriseGridSearchCV, train_test_split as surprise_train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Handle missing values and convert ratings to float
e_commerce['reviews.rating'] = pd.to_numeric(e_commerce['reviews.rating'], errors='coerce')
e_commerce = e_commerce.dropna(subset=['reviews.rating'])

# Combine features
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

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Content-Based Filtering
nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='auto')
nearest_neighbors.fit(tfidf_matrix_train)

# Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(e_commerce[['reviews.username', 'asins', 'reviews.rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# Hyperparameter tuning for SVD
param_grid = {
    'n_epochs': [20, 30, 40],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

grid_search = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
grid_search.fit(data)

# Extract the best parameters and retrain the SVD model
best_params = grid_search.best_params['rmse']
svd_best = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
svd_best.fit(trainset)

# Hybrid Recommender
def hybrid_recommend(user_id, item_id, num_recommendations=5):
    # Content-based part
    if item_id in train['asins'].values:
        item_idx = train[train['asins'] == item_id].index[0]
        distances, indices = nearest_neighbors.kneighbors(tfidf_matrix_train[item_idx], n_neighbors=num_recommendations+1)
        cb_recommendations = train.iloc[indices[0][1:]]['asins'].values
    else:
        cb_recommendations = []

    # Collaborative part
    cf_recommendations = []
    if user_id in train['reviews.username'].values:
        user_idx = train[train['reviews.username'] == user_id].index[0]
        user_ratings = train.iloc[user_idx]['reviews.rating']
        cf_recommendations = [train.iloc[i]['asins'] for i in user_ratings.argsort()[-num_recommendations:][::-1]]

    # Combine both
    hybrid_recommendations = list(set(cb_recommendations) | set(cf_recommendations))
    return hybrid_recommendations[:num_recommendations]

# Evaluation metrics
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
def evaluate_model(test_set, train_set, num_recommendations=5):
    precision_scores = []
    recall_scores = []
    true_ratings = []
    predicted_ratings = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        true_rating = test_set.iloc[idx]['reviews.rating']
        user_id = test_set.iloc[idx]['reviews.username']
        
        recommendations = hybrid_recommend(user_id, true_product_id, num_recommendations=num_recommendations)
        if not recommendations:
            continue
        
        recommended_ratings = [train[train['asins'] == rec]['reviews.rating'].mean() for rec in recommendations]

        y_true = [true_product_id]
        y_pred = recommendations

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

# Helper function to plot learning curves for Surprise models
def plot_surprise_learning_curve(algo, data, train_sizes=np.linspace(0.005, 0.1, 3), cv=5):
    train_errors, test_errors = [], []
    for train_size in train_sizes:
        train_rmse = []
        test_rmse = []
        for _ in range(cv):
            trainset, testset = surprise_train_test_split(data, test_size=1-train_size, random_state=42)
            algo.fit(trainset)
            train_pred = algo.test(trainset.build_testset())
            test_pred = algo.test(testset)
            train_rmse.append(accuracy.rmse(train_pred, verbose=False))
            test_rmse.append(accuracy.rmse(test_pred, verbose=False))
        train_errors.append(np.mean(train_rmse))
        test_errors.append(np.mean(test_rmse))

    plt.figure(figsize=(4, 3))
    plt.plot(train_sizes, train_errors, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_errors, 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curve (Surprise SVD)')
    plt.xlabel('Training size')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Helper function to plot validation curves for Surprise models
def plot_surprise_validation_curve(param_name, param_range, data, cv=5):
    param_grid = {param_name: param_range}
    gs = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse'], cv=cv, n_jobs=-1)
    gs.fit(data)

    test_scores_mean = [gs.cv_results['mean_test_rmse'][i] for i in range(len(param_range))]
    test_scores_std = [gs.cv_results['std_test_rmse'][i] for i in range(len(param_range))]

    plt.figure()
    plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(param_range, np.array(test_scores_mean) - np.array(test_scores_std),
                     np.array(test_scores_mean) + np.array(test_scores_std), alpha=0.1, color='g')
    plt.title(f'Validation Curve ({param_name})')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Define the plot_confusion_matrix function
def plot_confusion_matrix(y_true, y_pred):
    # Convert continuous ratings to discrete classes
    bins = [0, 1, 2, 3, 4, 5]  # Define your bins based on your rating scale
    y_true_binned = np.digitize(y_true, bins) - 1
    y_pred_binned = np.digitize(y_pred, bins) - 1
    
    cm = confusion_matrix(y_true_binned, y_pred_binned, labels=np.unique(y_true_binned))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true_binned))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Example Usage for the specific user and item
user_id = 'Cristina M'
item_id = 'B00QJDU3KY'
if item_id in e_commerce['asins'].values:
    recommendations = hybrid_recommend(user_id, item_id, num_recommendations=5)
    print("Recommendations for Product ID", item_id)
    print(recommendations)
else:
    print(f"Product ID {item_id} not found in the dataset.")

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall, rmse, rae, ndcg = evaluate_model(test, train, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RAE: {rae:.4f}")
print(f"NDCG@{num_recommendations}: {ndcg:.4f}")

# Plot confusion matrix
y_true = test['reviews.rating']
y_pred = [np.mean([train[train['asins'] == rec]['reviews.rating'].mean() for rec in hybrid_recommend(user_id, product_id, num_recommendations)]) for user_id, product_id in zip(test['reviews.username'], test['asins'])]
plot_confusion_matrix(y_true, y_pred)


# In[28]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, GridSearchCV as SurpriseGridSearchCV, train_test_split as surprise_train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Handle missing values and convert ratings to float
e_commerce['reviews.rating'] = pd.to_numeric(e_commerce['reviews.rating'], errors='coerce')
e_commerce = e_commerce.dropna(subset=['reviews.rating'])

# Combine features
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

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Content-Based Filtering
nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='auto')
nearest_neighbors.fit(tfidf_matrix_train)

# Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(e_commerce[['reviews.username', 'asins', 'reviews.rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# Hyperparameter tuning for SVD
param_grid = {
    'n_epochs': [20, 30, 40],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

grid_search = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
grid_search.fit(data)

# Extract the best parameters and retrain the SVD model
best_params = grid_search.best_params['rmse']
svd_best = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
svd_best.fit(trainset)

# Hybrid Recommender
def hybrid_recommend(user_id, item_id, num_recommendations=5):
    # Content-based part
    if item_id in train['asins'].values:
        item_idx = train[train['asins'] == item_id].index[0]
        distances, indices = nearest_neighbors.kneighbors(tfidf_matrix_train[item_idx], n_neighbors=num_recommendations+1)
        cb_recommendations = train.iloc[indices[0][1:]]['asins'].values
    else:
        cb_recommendations = []

    # Collaborative part
    cf_recommendations = []
    if user_id in train['reviews.username'].values:
        user_idx = train[train['reviews.username'] == user_id].index[0]
        user_ratings = train.iloc[user_idx]['reviews.rating']
        cf_recommendations = [train.iloc[i]['asins'] for i in user_ratings.argsort()[-num_recommendations:][::-1]]

    # Combine both
    hybrid_recommendations = list(set(cb_recommendations) | set(cf_recommendations))
    return hybrid_recommendations[:num_recommendations]

# Evaluation metrics
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
def evaluate_model(test_set, train_set, num_recommendations=5):
    precision_scores = []
    recall_scores = []
    true_ratings = []
    predicted_ratings = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        true_rating = test_set.iloc[idx]['reviews.rating']
        user_id = test_set.iloc[idx]['reviews.username']
        
        recommendations = hybrid_recommend(user_id, true_product_id, num_recommendations=num_recommendations)
        if not recommendations:
            continue
        
        recommended_ratings = [train[train['asins'] == rec]['reviews.rating'].mean() for rec in recommendations]

        y_true = [true_product_id]
        y_pred = recommendations

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

# Helper function to plot learning curves for Surprise models
def plot_surprise_learning_curve(algo, data, train_sizes=np.linspace(0.005, 0.1, 3), cv=5):
    train_errors, test_errors = [], []
    for train_size in train_sizes:
        train_rmse = []
        test_rmse = []
        for _ in range(cv):
            trainset, testset = surprise_train_test_split(data, test_size=1-train_size, random_state=42)
            algo.fit(trainset)
            train_pred = algo.test(trainset.build_testset())
            test_pred = algo.test(testset)
            train_rmse.append(accuracy.rmse(train_pred, verbose=False))
            test_rmse.append(accuracy.rmse(test_pred, verbose=False))
        train_errors.append(np.mean(train_rmse))
        test_errors.append(np.mean(test_rmse))

    plt.figure(figsize=(4, 3))
    plt.plot(train_sizes, train_errors, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_errors, 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curve (Surprise SVD)')
    plt.xlabel('Training size')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Helper function to plot validation curves for Surprise models
def plot_surprise_validation_curve(param_name, param_range, data, cv=5):
    param_grid = {param_name: param_range}
    gs = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse'], cv=cv, n_jobs=-1)
    gs.fit(data)

    test_scores_mean = [gs.cv_results['mean_test_rmse'][i] for i in range(len(param_range))]
    test_scores_std = [gs.cv_results['std_test_rmse'][i] for i in range(len(param_range))]

    plt.figure()
    plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(param_range, np.array(test_scores_mean) - np.array(test_scores_std),
                     np.array(test_scores_mean) + np.array(test_scores_std), alpha=0.1, color='g')
    plt.title(f'Validation Curve ({param_name})')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Define the plot_confusion_matrix function for binary format
def plot_confusion_matrix_binary(y_true, y_pred):
    # Convert continuous ratings to binary classes
    y_true_binary = [1 if rating >= 3 else 0 for rating in y_true]
    y_pred_binary = [1 if rating >= 3 else 0 for rating in y_pred]

    # Calculate the binary confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    # Plot the binary confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Example Usage for the specific user and item
user_id = 'Cristina M'
item_id = 'B00QJDU3KY'
if item_id in e_commerce['asins'].values:
    recommendations = hybrid_recommend(user_id, item_id, num_recommendations=5)
    print("Recommendations for Product ID", item_id)
    print(recommendations)
else:
    print(f"Product ID {item_id} not found in the dataset.")

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall, rmse, rae, ndcg = evaluate_model(test, train, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RAE: {rae:.4f}")
print(f"NDCG@{num_recommendations}: {ndcg:.4f}")

# Plot binary confusion matrix
y_true = test['reviews.rating']
y_pred = [np.mean([train[train['asins'] == rec]['reviews.rating'].mean() for rec in hybrid_recommend(user_id, product_id, num_recommendations)]) for user_id, product_id in zip(test['reviews.username'], test['asins'])]
plot_confusion_matrix_binary(y_true, y_pred)


# In[29]:


# Hyperparameter tuning for SVD
param_grid = {
    'n_epochs': [20, 30, 40],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

grid_search = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
grid_search.fit(data)

# Extract the best parameters and retrain the SVD model
best_params = grid_search.best_params['rmse']
svd_best = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
svd_best.fit(trainset)

# Helper function to plot learning curves for Surprise models
def plot_surprise_learning_curve(algo, data, train_sizes=np.linspace(0.1, 0.9, 5), cv=5):
    train_errors, test_errors = [], []
    for train_size in train_sizes:
        train_rmse = []
        test_rmse = []
        for _ in range(cv):
            trainset, testset = surprise_train_test_split(data, test_size=1-train_size, random_state=42)
            algo.fit(trainset)
            train_pred = algo.test(trainset.build_testset())
            test_pred = algo.test(testset)
            train_rmse.append(accuracy.rmse(train_pred, verbose=False))
            test_rmse.append(accuracy.rmse(test_pred, verbose=False))
        train_errors.append(np.mean(train_rmse))
        test_errors.append(np.mean(test_rmse))

    plt.figure(figsize=(7, 4))
    plt.plot(train_sizes, train_errors, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_errors, 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curve (Surprise SVD)')
    plt.xlabel('Training size')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Helper function to plot validation curves for Surprise models
def plot_surprise_validation_curve(param_name, param_range, data, cv=5):
    param_grid = {param_name: param_range}
    gs = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse'], cv=cv, n_jobs=-1)
    gs.fit(data)

    test_scores_mean = [gs.cv_results['mean_test_rmse'][i] for i in range(len(param_range))]
    test_scores_std = [gs.cv_results['std_test_rmse'][i] for i in range(len(param_range))]

    plt.figure(figsize=(7, 4))
    plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(param_range, np.array(test_scores_mean) - np.array(test_scores_std),
                     np.array(test_scores_mean) + np.array(test_scores_std), alpha=0.1, color='g')
    plt.title(f'Validation Curve ({param_name})')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Plot learning curves
plot_surprise_learning_curve(svd_best, data)

# Plot validation curves for n_epochs
param_range = np.arange(10, 110, 10)
plot_surprise_validation_curve('n_epochs', param_range, data, cv=5)

# Plot validation curves for lr_all
param_range = np.logspace(-4, -1, 4)
plot_surprise_validation_curve('lr_all', param_range, data, cv=5)

# Plot validation curves for reg_all
param_range = np.logspace(-3, 0, 4)
plot_surprise_validation_curve('reg_all', param_range, data, cv=5)


# In[30]:


import matplotlib.pyplot as plt

# Data for the plots
metrics = ['Precision@5', 'Recall@5', 'RMSE', 'RAE', 'NDCG@5']
values = [0.1780, 0.8898, 0.9096, 0.9145, 0.9030]

# Plotting the bar graph
plt.figure(figsize=(7, 4))

bars = plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])

# Adding text labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom')

# Adding title and labels
plt.title('Evaluation Metrics for Hybrid Filtering for product id B00QJDU3KY')
plt.ylabel('Values')

# Show the plot
plt.tight_layout()
plt.show()


# In[33]:


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.figure(figsize=(7, 3))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Plot learning curves
plot_surprise_learning_curve(svd_best, data)

# Plot validation curves for n_epochs
param_range = np.arange(10, 110, 10)
plot_surprise_validation_curve('n_epochs', param_range, data, cv=5)

# Plot validation curves for lr_all
param_range = np.logspace(-4, -1, 4)
plot_surprise_validation_curve('lr_all', param_range, data, cv=5)

# Plot validation curves for reg_all
param_range = np.logspace(-3, 0, 4)
plot_surprise_validation_curve('reg_all', param_range, data, cv=5)

# Example Usage
product_id = "B00QJDU3KY"
user_id = "example_user"
recommendations = get_recommendations(user_id, product_id, svd_best, cosine_similarity(tfidf_matrix_train), e_commerce)
print("Recommendations for Product ID", product_id)
print(recommendations)




# In[34]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV as SurpriseGridSearchCV, train_test_split as surprise_train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Handle missing values and convert ratings to float
e_commerce['reviews.rating'] = pd.to_numeric(e_commerce['reviews.rating'], errors='coerce')
e_commerce = e_commerce.dropna(subset=['reviews.rating'])

# Combine features
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

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Content-Based Filtering
nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='auto')
nearest_neighbors.fit(tfidf_matrix_train)

# Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(e_commerce[['reviews.username', 'asins', 'reviews.rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# Hyperparameter tuning for SVD
param_grid = {
    'n_epochs': [20, 30, 40],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

grid_search = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
grid_search.fit(data)

# Extract the best parameters and retrain the SVD model
best_params = grid_search.best_params['rmse']
svd_best = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
svd_best.fit(trainset)

# Helper function to plot learning curves for Surprise models
def plot_surprise_learning_curve(algo, data, train_sizes=np.linspace(0.1, 0.9, 5), cv=5):
    train_errors, test_errors = [], []
    for train_size in train_sizes:
        train_rmse = []
        test_rmse = []
        for _ in range(cv):
            trainset, testset = surprise_train_test_split(data, test_size=1-train_size, random_state=42)
            algo.fit(trainset)
            train_pred = algo.test(trainset.build_testset())
            test_pred = algo.test(testset)
            train_rmse.append(accuracy.rmse(train_pred, verbose=False))
            test_rmse.append(accuracy.rmse(test_pred, verbose=False))
        train_errors.append(np.mean(train_rmse))
        test_errors.append(np.mean(test_rmse))

    plt.figure(figsize=(5, 3))
    plt.plot(train_sizes, train_errors, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_errors, 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curve (Surprise SVD)')
    plt.xlabel('Training size')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Helper function to plot validation curves for Surprise models
def plot_surprise_validation_curve(param_name, param_range, data, cv=5):
    param_grid = {param_name: param_range}
    gs = SurpriseGridSearchCV(SVD, param_grid, measures=['rmse'], cv=cv, n_jobs=-1)
    gs.fit(data)

    test_scores_mean = [gs.cv_results['mean_test_rmse'][i] for i in range(len(param_range))]
    test_scores_std = [gs.cv_results['std_test_rmse'][i] for i in range(len(param_range))]

    plt.figure(figsize=(5, 3))
    plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(param_range, np.array(test_scores_mean) - np.array(test_scores_std),
                     np.array(test_scores_mean) + np.array(test_scores_std), alpha=0.1, color='g')
    plt.title(f'Validation Curve ({param_name})')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.show()

# Define the plot_confusion_matrix function
def plot_confusion_matrix(y_true, y_pred, labels=[0, 1], figsize=(3, 2)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# Hybrid Recommender with cold start handling
def hybrid_recommend(user_id, item_id, model_cf, cosine_sim_matrix, items_data, num_recommendations=5):
    # Cold start for new item
    if item_id not in items_data['asins'].values:
        return items_data['asins'].value_counts().index[:num_recommendations].tolist()

    # Content-based part
    item_idx = items_data[items_data['asins'] == item_id].index[0]
    if item_idx >= tfidf_matrix_train.shape[0]:  # Use .shape[0] instead of len()
        return []
    distances, indices = nearest_neighbors.kneighbors(tfidf_matrix_train[item_idx], n_neighbors=num_recommendations+1)
    cb_recommendations = items_data.iloc[indices[0][1:]]['asins'].values

    # Cold start for new user
    if user_id not in items_data['reviews.username'].values:
        return cb_recommendations[:num_recommendations]

    # Collaborative part
    cf_recommendations = []
    user_idx = items_data[items_data['reviews.username'] == user_id].index
    if user_idx.empty:
        return cb_recommendations[:num_recommendations]
    user_ratings = items_data.loc[user_idx[0]]['reviews.rating']
    cf_recommendations = [items_data.iloc[i]['asins'] for i in user_ratings.argsort()[-num_recommendations:][::-1]]

    # Combine both
    hybrid_recommendations = list(set(cb_recommendations) | set(cf_recommendations))
    return hybrid_recommendations[:num_recommendations]

# Evaluation metrics
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
def evaluate_model(test_set, train_set, model_cf, cosine_sim_matrix, num_recommendations=5):
    precision_scores = []
    recall_scores = []
    true_ratings = []
    predicted_ratings = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        true_rating = test_set.iloc[idx]['reviews.rating']
        user_id = test_set.iloc[idx]['reviews.username']
        
        recommendations = hybrid_recommend(user_id, true_product_id, model_cf, cosine_sim_matrix, train_set, num_recommendations)
        if len(recommendations) == 0:
            continue
        
        recommended_ratings = [train_set[train_set['asins'] == rec]['reviews.rating'].mean() for rec in recommendations]

        y_true = [true_product_id]
        y_pred = recommendations

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

# Example Usage for cold start problem
product_id = "B00QJDU4KY"
user_id = "new_user"  # Assume this is a new user
recommendations = hybrid_recommend(user_id, product_id, svd_best, cosine_similarity(tfidf_matrix_train), e_commerce, num_recommendations=5)
print("Recommendations for Product ID", product_id)
print(recommendations)

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall, rmse, rae, ndcg = evaluate_model(test, train, svd_best, cosine_similarity(tfidf_matrix_train), num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RAE: {rae:.4f}")
print(f"NDCG@{num_recommendations}: {ndcg:.4f}")

# Plot the evaluation metrics
metrics = {
    'Precision': avg_precision,
    'Recall': avg_recall,
    'RMSE': rmse,
    'RAE': rae,
    'NDCG': ndcg
}

plt.figure(figsize=(5, 3))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Evaluation Metrics for Hybrid Recommender System')
plt.ylim(0, 1.2)  # Set y-axis limit to highlight the differences
for index, value in enumerate(metrics.values()):
    plt.text(index, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
plt.show()

# Convert ratings to integer values if they are not already
test['reviews.rating'] = test['reviews.rating'].astype(int)
train['reviews.rating'] = train['reviews.rating'].astype(int)

# Function to map ratings to binary classification for simplicity
def rating_to_class(rating):
    return 1 if rating >= 3 else 0

# Apply the function to get binary classes
y_true = test['reviews.rating'].apply(rating_to_class)
y_pred = []

for user_id, product_id in zip(test['reviews.username'], test['asins']):
    recommendations = hybrid_recommend(user_id, product_id, svd_best, cosine_similarity(tfidf_matrix_train), e_commerce, num_recommendations)
    if len(recommendations) == 0:
        y_pred.append(0)  # Assuming 0 for no recommendation
    else:
        avg_rating = np.mean([train[train['asins'] == rec]['reviews.rating'].mean() for rec in recommendations])
        y_pred.append(rating_to_class(avg_rating))

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Plot confusion matrix
#plot_confusion_matrix(y_true, y_pred)
# Plot confusion matrix with reduced size
plot_confusion_matrix(y_true, y_pred, labels=np.unique(y_true), figsize=(3, 2))

# Plot learning curves
plot_surprise_learning_curve(svd_best, data)

# Plot validation curves for n_epochs
param_range = np.arange(10, 110, 10)
plot_surprise_validation_curve('n_epochs', param_range, data, cv=5)

# Plot validation curves for lr_all
param_range = np.logspace(-4, -1, 4)
plot_surprise_validation_curve('lr_all', param_range, data, cv=5)

# Plot validation curves for reg_all
param_range = np.logspace(-3, 0, 4)
plot_surprise_validation_curve('reg_all', param_range, data, cv=5)


# In[35]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Assuming e_commerce dataframe is already loaded
# Handle missing values and convert ratings to float
e_commerce['reviews.rating'] = pd.to_numeric(e_commerce['reviews.rating'], errors='coerce')
e_commerce = e_commerce.dropna(subset=['reviews.rating'])

# Combine features
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

# Split into training and testing sets
train, test = train_test_split(e_commerce, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['combined_features'])
tfidf_matrix_test = tfidf_vectorizer.transform(test['combined_features'])

# Content-Based Filtering
nearest_neighbors = NearestNeighbors(metric='cosine', algorithm='auto')
nearest_neighbors.fit(tfidf_matrix_train)

# Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(e_commerce[['reviews.username', 'asins', 'reviews.rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# SVD model with specific parameters
svd_best = SVD(n_epochs=60, lr_all=0.02, reg_all=0.02)
svd_best.fit(trainset)

# Hybrid Recommender
def hybrid_recommend(user_id, item_id, num_recommendations=5):
    # Content-based part
    if item_id in train['asins'].values:
        item_idx = train[train['asins'] == item_id].index[0]
        distances, indices = nearest_neighbors.kneighbors(tfidf_matrix_train[item_idx], n_neighbors=num_recommendations+1)
        cb_recommendations = train.iloc[indices[0][1:]]['asins'].values
    else:
        cb_recommendations = []

    # Collaborative part
    cf_recommendations = []
    if user_id in train['reviews.username'].values:
        user_idx = train[train['reviews.username'] == user_id].index[0]
        user_ratings = train.iloc[user_idx]['reviews.rating']
        cf_recommendations = [train.iloc[i]['asins'] for i in user_ratings.argsort()[-num_recommendations:][::-1]]

    # Combine both
    hybrid_recommendations = list(set(cb_recommendations) | set(cf_recommendations))
    return hybrid_recommendations[:num_recommendations]

# Evaluation metrics
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
def evaluate_model(test_set, train_set, num_recommendations=5):
    precision_scores = []
    recall_scores = []
    true_ratings = []
    predicted_ratings = []

    for idx in range(len(test_set)):
        true_product_id = test_set.iloc[idx]['asins']
        true_rating = test_set.iloc[idx]['reviews.rating']
        user_id = test_set.iloc[idx]['reviews.username']
        
        recommendations = hybrid_recommend(user_id, true_product_id, num_recommendations=num_recommendations)
        if not recommendations:
            continue
        
        recommended_ratings = [train[train['asins'] == rec]['reviews.rating'].mean() for rec in recommendations]

        y_true = [true_product_id]
        y_pred = recommendations

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

# Plot binary confusion matrix
def plot_confusion_matrix_binary(y_true, y_pred):
    y_true_binary = [1 if rating >= 3 else 0 for rating in y_true]
    y_pred_binary = [1 if rating >= 3 else 0 for rating in y_pred]

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Binary Confusion Matrix')
    plt.show()

# Example Usage for a specific user and item
user_id = 'Cristina M'
item_id = 'B00QJDU3KY'
if item_id in e_commerce['asins'].values:
    recommendations = hybrid_recommend(user_id, item_id, num_recommendations=5)
    print("Recommendations for Product ID", item_id)
    print(recommendations)
else:
    print(f"Product ID {item_id} not found in the dataset.")

# Evaluate the model
num_recommendations = 5
avg_precision, avg_recall, rmse, rae, ndcg = evaluate_model(test, train, num_recommendations)
print(f"Average Precision at {num_recommendations}: {avg_precision:.4f}")
print(f"Average Recall at {num_recommendations}: {avg_recall:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RAE: {rae:.4f}")
print(f"NDCG@{num_recommendations}: {ndcg:.4f}")

# Plot binary confusion matrix
y_true = test['reviews.rating']
y_pred = [np.mean([train[train['asins'] == rec]['reviews.rating'].mean() for rec in hybrid_recommend(user_id, product_id, num_recommendations)]) for user_id, product_id in zip(test['reviews.username'], test['asins'])]
plot_confusion_matrix_binary(y_true, y_pred)


# In[ ]:




