import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import hebrew_tokenizer as ht
import joblib
import numpy as np
from helpers.constants import ISRAELI_DATA_PATH, MACRO_NUTRIENTS, MICRO_NUTRIENTS, NUMBER_OF_FOOD_GROUPS, FOOD_STOP_WORDS

CUTOFF = 30
# Libraries for Barel: Pandas, sklearn, hebrew_tokenizer, joblib





def preprocess_data():
    data = pd.read_csv(ISRAELI_DATA_PATH)
    data = data[['shmmitzrach'] + MACRO_NUTRIENTS + ['alcohol'] + MICRO_NUTRIENTS]

    # Applying chosen clustering for evaluation:
    for macro in MACRO_NUTRIENTS + ['alcohol']:
        data[macro] = data[macro].fillna(0)  # in order to avoid index miss-match when recommending
    clusters = AgglomerativeClustering(n_clusters=NUMBER_OF_FOOD_GROUPS, affinity="manhattan", linkage="complete")
    clusters.fit(data[MACRO_NUTRIENTS + ['alcohol']])
    data['label'] = clusters.labels_

    # Loading micro-nutrient predictions saved model in order to predict NaN values
    model = joblib.load("prediction.joblib")
    predictions = model.predict(data[MACRO_NUTRIENTS])
    for i, micro in enumerate(MICRO_NUTRIENTS):
        data[f'{micro}_pred'] = predictions[:, i]
        data[micro] = np.where(data[micro] >= 0, data[micro], data[micro+"_pred"])
    return data


def tokenize(hebrew_text):
    tokens = ht.tokenize(hebrew_text)  # tokenize returns a generator!
    words = [
        token
        for grp, token, token_num, (start_index, end_index) in tokens
        if grp == 'HEBREW'
    ]

    return words


def find_food_item(data, item):
    food_item = item  # Example: חלב 3% שומן, תנובה, טרה, הרדוף, יטבתה
    all_food_names = data['shmmitzrach'].squeeze()
    food_found = False
    if food_item in all_food_names.values:
        print("found food item")
        food_found = True
    else:
        print("did not find food item... looking for closest match")
        for name in all_food_names.values:
            if food_item in name:
                food_item = name
                food_found = True
                break
    if not food_found:
        food_item = -1
    return food_item


def get_recommendations(data, food_item, cosine_sim, indices, k=10):
    # Get the index of the food item
    idx = indices[food_item]

    # Get the pairwsie similarity scores of all foods with that food item
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the food items based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the k most similar food items
    sim_scores = sim_scores[1:k+1]

    # Get the food items indices
    foods_indices = [i[0] for i in sim_scores]

    # Return the top k most similar food items
    return data['shmmitzrach'].iloc[foods_indices]


def content_based_recommender(item):
    data = preprocess_data()
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=FOOD_STOP_WORDS)
    tfidf_matrix = tfidf.fit_transform(data['shmmitzrach'])
    cosine_sim_names = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim_macros = cosine_similarity(data[MACRO_NUTRIENTS])
    cosine_sim_micros = cosine_similarity(data[MACRO_NUTRIENTS+MICRO_NUTRIENTS])
    indices = pd.Series(data.index, index=data['shmmitzrach']).drop_duplicates()
    food_item = find_food_item(data, item)
    return data, food_item, cosine_sim_names, cosine_sim_macros, cosine_sim_micros, indices


def show_recommendations(item):
    data, food_item, cosine_sim_names, cosine_sim_macros, cosine_sim_micros, indices = content_based_recommender(item)
    if food_item == -1:
        return None, None, None, food_item
    first_recommendation = get_recommendations(data, food_item, cosine_sim_names, indices)
    
    second_recommendation = get_recommendations(data, food_item, cosine_sim_macros, indices)
    
    third_recommendation = get_recommendations(data, food_item, cosine_sim_micros, indices)
    
    return first_recommendation, second_recommendation, third_recommendation, food_item


