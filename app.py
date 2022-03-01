import io
from optparse import Values
import os
from re import X
from turtle import width
from typing import List
from helpers.constants import MACRO_NUTRIENTS
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import plotly.express as px
from recommender_system import show_recommendations
from PIL import Image

__DIRNAME__ = os.path.dirname(os.path.realpath(__file__))
DATASET = pd.read_csv(os.path.join(__DIRNAME__, 'israeli_data.csv'))

FOOD_GROUP_EXAMPLE = {
    0: [56208068,56200410,50020000,57602118,56113010,56207258,56200500,51401000,56205110],# הדגנים
    1: [75117010,73101010,75109000,73201040,73302020,90000065,75111000,75111030,75102500],# הירקות
    2: [63139010,63135010,61101010,63107010,63115010,63146010,63126500,63126510,63109010],# פירות
    3: [41302020,41303030,41101120,41400030,75224022,41400030,41205149,41209028,42116030],# הקטניות
    4: [43102000,43102119,43101000,42107000,82104000,82101000,82101029,82101100,82101300],# קבוצת השמנים
}

st.set_page_config(layout="wide")
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def recommender():
  # taking only the items names 
  data = DATASET[['shmmitzrach']].rename(columns={'shmmitzrach': 'שם מצרך'})

  # write into the app
  st.markdown('**FOOD NAMES IN DATASET**')
  st.write(data)
  
  item = st.text_input('כדי לרשום שם של מוצר מזון כדי לקבל המלצה למשל חסה: ')
  if item != '':
    recommend(item)
  return 


def select_word_cloud():
    path = os.path.join(__DIRNAME__, 'results', 'word_cloud')
    
    photos_name = os.listdir(path)

    to_img = lambda x: Image.open(os.path.join(__DIRNAME__, 'results', 'word_cloud', x))

    dic_imagens = {f'Lable {x}': to_img(x) for x in photos_name}
    
    select_box(dic_imagens)

def make_parallel_coordinates(df, color, columns):
    return px.parallel_coordinates(
        df[columns], 
        color=color,
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2, width=1000)


def get_data_for_parallel_coordinates(df: pd.DataFrame, columns):
    fig = make_parallel_coordinates(
        df, 
        columns[0], 
        columns
        )
    return st.plotly_chart(fig)


def select_box(dicty):
    pic = st.selectbox("Title", list(dicty.keys()))
    st.image(dicty[pic], use_column_width=True)


def recommend(item):
  r1, r2, r3, food_item = show_recommendations(item)
  if r1 is None:
      ''' **Could not find your item please try another** '''
      return 
  group1 = pd.Series([1 for _ in range(r1.size)], name='recommendation')
  group2 = pd.Series([2 for _ in range(r2.size)], name='recommendation')
  group3 = pd.Series([3 for _ in range(r3.size)], name='recommendation')
  r1, r2, r3 = pd.concat([r1.reset_index(drop=True), group1], axis=1), pd.concat([r2.reset_index(drop=True), group2], axis=1 ), pd.concat([r3.reset_index(drop=True), group3], axis=1)
  df = pd.concat([r1, r2, r3])
  df_foods_items = DATASET.set_index('shmmitzrach').loc[df.shmmitzrach.values]
  df_foods_items['three_type_of_recommendtion'] = df.recommendation.values
  columns = ['three_type_of_recommendtion', 'protein', 'total_fat', 'carbohydrates']
  if food_item != item:
      f'''
      **Could not find your chosen food item in the data.**
      
      Showing recommendtions for the first match - {food_item}.

      For exact match look at the data above
      '''
  else:
      f'Showing recommendtions for {food_item}'  


  st.markdown(''' #####  המלצה לפי טקסט דומה ''')
  r1
  st.markdown(''' #####  המלצה לפי ערכי מאקרו דומים ''')
  r2
  st.markdown(''' ##### המלצה לפי ערכי מאקרו ומיקרו דומים ''')
  r3

  get_data_for_parallel_coordinates(df_foods_items.reset_index(), columns)

def getnut():
    nut = Image.open(os.path.join(__DIRNAME__, 'results', 'nutpng.png'))
    st.image(nut, width=100)

def rainbow():
    nut = Image.open(os.path.join(__DIRNAME__, 'results', 'the-nutritional-rainbow.jpg'))
    st.image(nut, width=500)

def parallel_coordinates():   
    df_foods_items = DATASET.set_index('smlmitzrach').loc[[item for items in FOOD_GROUP_EXAMPLE.values() for item in items]]

    food_group_types = [[group]*len(Values) for group, Values in FOOD_GROUP_EXAMPLE.items()]
    flatten = [groupid for group in food_group_types for groupid in group]
    print(type(flatten[0]))
    df_foods_items['TypeOfFoodGroup'] = flatten
    

    columns = ['TypeOfFoodGroup', 'protein', 'total_fat', 'carbohydrates', 'calcium', 'iron', 'magnesium']

    
    st.plotly_chart(px.parallel_coordinates(
        df_foods_items[columns], 
        color='TypeOfFoodGroup',
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2, width=1000))

def get_radar():
    st.subheader('Figure F1')
    radar = Image.open(os.path.join(__DIRNAME__, 'results', 'radar.png')) #learning_algo_accuracy.jpg
    st.image(radar, use_column_width=True)




def main():
    # lottie_book = load_lottieurl('https://assets2.lottiefiles.com/temp/lf20_nXwOJj.json')
    # st_lottie(lottie_book, speed=1, height=200, key="initial")
    one, center, two = st.columns((1, 2, 1))
    with center:
        rainbow()

    one, row0_1, three, row0_2, two = st.columns((.1, 2, 0.5, 1, .1))
    row0_1.title('NUT - Nutritionist Utility Tool')
    with three:
        getnut()
    


    with row0_2:
        st.write('')

    row0_2.subheader('A Streamlit web app by Barel Mishal, Sapir Shapira, Yoav Orenbach and Hen Emuna')
    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

    with row1_1:
        st.header('**Motivation and problem description:**')
        st.markdown("Food and Nutrition Science is a newly developed field critical for health and development. It affects our body’s homeostasis in general, specifically our immune system, sugar balance, and endocrine system. As a young research field, there is a need to deepen the field’s knowledge further. Moreover, we cannot partially leverage existing knowledge, as it contains many myths and biases, some of which are incorrect. An excellent example of this bias is the relation to fat, which is incorrectly considered unhealthy. In fact, it is the vast intake of sugar that comes from this thinking that causes many of the diseases today. As a result, Nutritionists must use both reliable information and scientifically supported tools to know the exact composition of food. This knowledge can be used both for personal diet recommendation as well as interdisciplinary research that may link the different nutritions to metabolic processes and a wide range of non-communicable diseases: type 2 diabetes, developmental issues, etc [1].")
        st.markdown("A critical aspect of daily nutritionist’s work is to suggest dietary alternatives for patients, based on their specific health profile, using previously defined food groups (s.a., Meat, milk, vegetables, fruits, and sweets). The rationale behind the clustering to food groups,  based on known macronutrients (carbohydrates, fats, and proteins) and an estimation of the equivalent micronutrients (Vitamins, and minerals), is to enable the creation of reliable tools, which would provide nutritionists and their patients the ability to create dietary alternatives and control regarding their food consumption. Nowadays, this clustering relies mainly on assumptions and lists of products memorized by Nutritionists.")

    row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
    with row2_1:
        st.header('**Data:**')

        st.markdown("We gathered from the [Israeli government database catalog](https://info.data.gov.il/datagov/home/) a table (10MBs) with 4650 records of 74 [nutritional components](https://data.gov.il/dataset/nutrition-database): protein, fats, carbohydrates, amino acids, fatty acids, vitamins, and minerals. The table was created from a JSON file containing records and metadata, which was pre-processed (units conversion, missing fields, dropping non-helpful features like psolet, Supplements, and Recipes) and saved as a CSV. In addition, since the Israeli data quite is small we also use the U.S department of agriculture Food Data Central (FDC) dataset (135MBs) which the Israeli data relies on, in order to attempt to improve some of our results using more data. After using the FDC API key to extract information (with web scraping), we built a table of 7600 records of 149 nutritional components containing the components of the 1st table.")
        st.markdown("Comparing the two, the FDC dataset has more records than the Israeli data as well as additional features per record (for example, FDC uses 3 types of vitamin K, whereas the Israeli data uses just one type based on those three types). Therefore, while these datasets are connected, their features can’t be mapped easily, in terms of features and item ids, requiring preprocessing to match them to the best of our knowledge.")
        st.markdown('''
        - Israeli nutrition data source (https://data.gov.il/dataset/nutrition-database) from the ministry of health.
        - Food Data Central source (https://fdc.nal.usda.gov/)
        ''')

    row2_spacer1, row_3, row2_spacer2 = st.columns((.1, 3.2, .1))

    with row_3:
        st.header('**Solution:**')
        st.markdown("Our goal is to improve the existing assumptions and biases used in the Israeli nutrition community, thus helping nutritionists offer better alternatives for their patience. In particular, we intend to make improvements in three areas:")

        st.subheader('**1.** Clustering of food items to food groups.')
        st.markdown('''There are no known food group labels for food items in the Israeli data (or the American data for that matter) since some food items cannot be classified into one food group but into several. For example, if we look at a food item - Pizza, it could fall into the Fat food group as it has high amounts of fat, however, it could also fall into the dairy food group depending on the amount of cheese on the pizza. Moreover, a pizza could be in the pastries food group because it contains bread. Many more examples follow, though there are some food items which we unequivocally know their food group, like milk, for instance.''')
        st.markdown("Therefore, our objective as a first step in the analysis is to cluster all food items in the Israeli data into their respective food group.") 
        st.markdown('''
        To deal with the ambiguity in the food-groups clustering, we received a [list from the ministry of health](https://docs.google.com/document/d/1aJYJs4XUrMqhAfBYMaFdSmwESYzJKPk9/edit?usp=sharing&ouid=108062365532375633132&rtpof=true&sd=true) containing all the Israeli food groups with known food items in each group (containing 11 main food groups and 32 sub food groups). However, this list is lacking, containing only 320 food items. To expand our food-groups estimation with this ground-truth information, we use a set of clustering algorithms to cluster the additional food items in our dataset, i.e. the missing food items in the given list, as close as we can to the known food items.
        For our clustering algorithm, we compared the following (algorithms with varying un/even cluster size, different expected manifolds geometry, with/out outlier removal):

- K Means
- Agglomerative
- DBScan
- Spectral clustering.

        ''') 
        get_radar()
        st.markdown('Comparing the proportion of the three macronutrients (proteins, carbohydrates, and fats) and alcohol for different food items; Chicken breast (blue) belongs to Meat group for chicken and turkey, Pita (red) belongs to Pastry group for breads, Hazelnut (green) belongs to Nuts and seeds group, and Wine (purple) belongs to Alcoholic beverages.')
        # parallel_coordinates()
        st.markdown("Visualization of our clustering result can be seen at the following  [Heroku dashboard site](https://dashboard-food-group.herokuapp.com/) site (might take a few seconds to load), allowing a qualitative evaluation of the clustering, for each of the chosen algorithms (a quantitative evaluation is expanded under ‘evaluation’ section).")
        st.markdown('''
        Our clustering results are verified between the different algorithms used. Fig 2 shows an example of a Spectral clustering result, where we map each cluster to the matching food-group, using the known food-items list mentioned above.
        
        As we can see, there are cases where the clustering result is not ideal, meaning we don’t get a one-to-one mapping between each cluster and its respective food group, but rather several food-groups (at most 3 food groups assigned).

        ''')

        st.markdown('Mapping food groups to the clustering algorithm`s labels. Green checkmark signifies a mapping between a food group and a label. An ideal  clustering would have a one-to-one mapping between the food groups and the labels. As we can see there are mismatches in this mapping, for example cluster #24 has two contradicting food groups, and cluster #8 does not match any known food groups.')
        get_radar() # TODO: switch it to mapping table

        st.subheader('**2.** Predicting food’s micronutrients based on their macronutrients using machine and deep learning principles.')
        st.markdown('''Our input contains the macronutrients – proteins, carbohydrates and fats, while our output contains the micronutrients – vitamins and minerals.
    We applied many machine-learning algorithms since the connection between macronutrients and micronutrients is not pre-established. Specifically we used:
    Linear regression (compared with variations s.a. ridge regression and kernels), K-nearest neighbors, Gaussian-process regression, Tree-based regressors (with ensembles): Decision tree, Random forest, XGBoost, Neural network (multi-layer perceptron)''')

    row2_spacer1, row_7, row2_spacer2 = st.columns((.1, 3.2, .1))
    with row_7:
        st.subheader('**3.** Recommendation of food alternatives based on similar products within the same food group.')

        st.markdown(
            '''
Since there are no known ratings and we did not have enough time to gather ratings for the food items in our data, we decided to use a content based recommender system. One of the downsides of using content based recommenders is that finding the appropriate features for each food item profile is hard. Therefore, we used three different feature vectors:

a)    Food items names

b)    Food items macronutrients

c)    Food items macronutrients + micronutrients

As for the prediction heuristic, given a food item and any of the item’s profiles, we compute the cosine similarity between them and return the top ten most similar food items (based on that profile). In addition, since many micronutrients values are missing from the data and we wish to use them for our third feature vector, we use our second step – the micronutrient prediction to fill any missing values. ''')

    
    line1_spacer1, line1_1, line1_spacer2 = st.columns((.1, 3.2, .1))

    with line1_1:
        st.header('Evaluation:')  
        st.markdown('Evaluation criteria, results and visualizations for each part:')
    
    
    line1_spacer1, line2_1, line1_spacer2 = st.columns((.1, 3.2, .1))

    with line2_1:
        st.subheader('1.    Evaluating food groups cluster:')

        st.markdown(
                '''
    Seeing as there are no labels of food groups, we opted to use domain knowledge to evaluate our clustering. As noted before, we received a list from the ministry of health containing the known food items of each one of the 32 food groups and used this knowledge for our evaluation, however, the food items in it do not perfectly match the Israeli data, causing a problem with such a list.

    Hence, we first mapped 32 food items from the data into their respective food group. Then, after applying a clustering algorithm, we mapped the cluster labels of the 32 food items to the food groups we already assigned. For instance, if food item 1 was mapped to the meat food group and the clustering algorithm put it in the third cluster (so cluster label 3), then label 3 maps to the meat food group.

    If we manage to receive one-to-one mapping between the algorithm’s clustering labels to the food groups (so every label has a single food group), we can consider the clustering as a success. However, if a food group was given more than one label, then we know the clustering was not perfect, as we know from the ministry of health that the food items we used belong to different food groups.

    The results we got varied between different clustering algorithms, yet Spectral clustering achieved the most dispersed labels with at most 3 food groups assigned to a label.

    We visualize this result using a table where a green checkmark signifies a mapping between a food group and a label:
    ''')
        
        st.image(Image.open(os.path.join(__DIRNAME__, 'results', 'Mappingtable.jpg')))
        
        st.markdown('''
    This is the table received for Spectral clustering and all tables can be seen at the Heroku dashboard site.
        
    The main issue with this evaluation is that it cannot be quantified as there are no real ground truths we can use. Thus, in order to see a numerical evaluation of our clustering, we created a new table containing all the known food items of every food group from our data (which came out to 320 food items) using the items from the ministry of health list. We applied our clustering algorithms to those food items and compared the resulting clustering labels to the ground truths using the clustering metrics: Adjusted Rand Index score, Fowlkes Mellows score, Adjusted Mutual Info score and the V-measure score (All of the above are measures for comparing two clusters).

    Using these metrics we could further optimize our clustering algorithms (for instance, we decided to also use alcohol as part of our input as it also helped us increase our scores), and the scores each algorithm achieved are the following:
        ''')

    one, two = st.columns((1, 1)) 

    with one:
        path = os.path.join(__DIRNAME__, 'results', 'results_clustering_comparison_flip.jpg')
        st.image(Image.open(path), use_column_width=True)
    with two:
        path2 = os.path.join(__DIRNAME__, 'results', 'results_clustering_comparison.jpg')
        st.image(Image.open(path2), use_column_width=True)     

    line1_spacer1, line2_13, line1_spacer2 = st.columns((.4, 2, .4))

    with line2_13:
        st.markdown('''
We can see that the Agglomerative clustering achieves the best scores on all metrics, though its mapping table is not as good as the Spectral clustering table. The results are still lacking, however, some sub food groups have very similar macronutrient breakdown, so it is quite hard to cluster them perfectly when the inputs are very similar. Finally, we can see that the clustering does a gfood job with many food groups and show it with a word cloud of for each food group:
''')
        select_word_cloud()        

    line1_spacer1, line2_1, line1_spacer2 = st.columns((.1, 3.2, .1))

    with line2_1:

        st.subheader('2. Evaluating micronutrients prediction:')

        st.markdown(
            '''
    The prediction task is a multi-output regression problem, thus the straightforward way to evaluate it is by splitting our data into train and test sets and computing our accuracy on the test set. However, we are dealing with numbers in the Microgram or Nano-gram ranges, so it was proven quite difficult to achieve high accuracy with exact numbers. Therefore, we wanted to know how far away we are from the exact micronutrient value. For this reason, when calculating the accuracy we decided to give a small error range of one micro-gram and we can consider our predictive model to be successful if it is correct with an error on one microgram.
            
    The main issue we had with this evaluation is the lack of data (we only have 4560 food items). Therefore we decided to use the FDC data as part of our training data, and since we want to be successful with the Israeli data, our test set only contained food items from there.
    After Applying all regression algorithms and taking into account a small error of one Microgram the accuracy results we got are:

            ''')

        path2 = os.path.join(__DIRNAME__, 'results', 'accu_on_test.jpg')
        st.image(Image.open(path2), use_column_width=True)   

        st.markdown(
            '''
    Firstly, we can see that the FDC data does not help much or even at all, which is probably due to overfitting to it or because it is still not enough to properly learn (even combined, the number of food items is just a little over 10,000). Secondly, we can see that the Decision Tree algorithm achieved the highest accuracy with about 71% accuracy on average. It does not seem like much, which is why we tried exploring applying the same algorithms for just one micro-nutrient at a time (instead of a multi-output). We saw that some algorithms perform better than others for different micronutrients. For instance, we saw that for vitamin b6 the Linear regression algorithm performs best with 98% accuracy, so we assume this vitamin has a linear relation to the macronutrients, while other vitamins and minerals have different relations. In conclusion, we see that on average the decision tree algorithm is the most fit for the prediction task, achieving very high accuracy on some micro-nutrients and also lower on others depending on their relation.           
            ''')

        st.write('')

        st.subheader('3. Evaluating food alternatives recommendation:')
        
        
        recommender()


        # st.pyplot(fig)

        st.markdown(
            '''
Due to the fact there are no ratings of food items we can use, we cannot evaluate our recommender system by creating a test set and computing the root-mean-square error for example. However, there is a way for us to know if our recommender is indeed successful and that is by using the Precision and the Normalized Discounted cumulative gain (NDCG) evaluation metrics.

Given a food item, for each feature vector we recommend K items which we consider as positives and in order to calculate Precision and NDCG, we use our first step – the food groups clustering. Since we want to offer alternative food items within the same food group, we compare the food group label of the given food item to the food group label of the recommended items. For the K positive recommendations, if the labels match than we consider it a true positive, otherwise it is a false positive. With that in mind, we can compare the different feature vectors of the recommendations on different food items to see which has the best results, and also find the proper value of K to see how many items we should recommend.

The result we got for 2 food item (milk and lettuce) are:
            ''')

    one, two = st.columns((1, 1)) 

    with one:
        path = os.path.join(__DIRNAME__, 'results', '1NDCG.jpg')
        st.image(Image.open(path), use_column_width=True)
    with two:
        path2 = os.path.join(__DIRNAME__, 'results', '2NPRE.jpg')
        st.image(Image.open(path2), use_column_width=True)   
    with one:
        path = os.path.join(__DIRNAME__, 'results', '3NDCG.jpg')
        st.image(Image.open(path), use_column_width=True)
    with two:
        path2 = os.path.join(__DIRNAME__, 'results', '4PRECI.jpg')
        st.image(Image.open(path2), use_column_width=True)    

    line1_spacer1, line2_1, line1_spacer2 = st.columns((.1, 3.2, .1))


    with line2_1:   
        st.markdown(
            '''
We can see that for the food item milk we achieve very high NDCG for the macronutrient and micronutrient feature vector, while for the food item lettuce we achieve high NDCG for the name feature vector. Similar cases follow, and while we cannot show the results for every food item, we can see that on average K=10 has the best scores for both Precision and NDCG, and because some feature vectors perform better than others for different food items, we show 3 different recommendations. Specifically, in the streamlit app, one can enter a food item in Hebrew and see 3 different kinds of recommendation based on the three item profiles.
            ''')
 
    line1_spacer1, line2_12, line1_spacer2 = st.columns((.1, 3.2, .1))

    with line2_12:    
        
        st.subheader('3. Evaluating food alternatives recommendation:')

        st.markdown('''
In the future we believe an improved clustering can be achieved by a better filtering of outliers and noise (food items that should not be clustered to any food group like ‘similak’). While we tried our best to filter any noise, the results we got are still lacking and we believe that a better constructed data can help in the food groups clustering.

Moreover, we believe further investigation regarding the relation of every micronutrient to the macronutrients can help in deciding which algorithm should be used for every micro-nutrient and improve the prediction accuracy.

Finally, by collecting user data on the Israeli food items, the recommendation can be improved significantly and with enough ratings, a collaborative recommender system could be used.
            ''')

    

    line1_spacer1, line2_12, line1_spacer2 = st.columns((.1, 3.2, .1))

    with line2_12:    
        
        st.subheader('Future Work:')

        st.markdown('''
In the future we believe an improved clustering can be achieved by a better filtering of outliers and noise (food items that should not be clustered to any food group like ‘similak’). While we tried our best to filter any noise, the results we got are still lacking and we believe that a better constructed data can help in the food groups clustering.

Moreover, we believe further investigation regarding the relation of every micronutrient to the macronutrients can help in deciding which algorithm should be used for every micro-nutrient and improve the prediction accuracy.

Finally, by collecting user data on the Israeli food items, the recommendation can be improved significantly and with enough ratings, a collaborative recommender system could be used.
''')

        st.subheader('Conclusion:')

        st.markdown('''
In this project we got to experiment with different areas of Israeli nutrition, attempting to help nutritionist better classify the food groups of food items, predict unknown micronutrients, rather than relying on intuition and hopefully offer better food item alternatives for their patients.

We hope our findings can help the nutrition field and act as a basis for further improvements.
''')

        st.header('')

        st.header('Bibliography:')
        # st.pyplot(fig)
        st.markdown('''
    1. Mozaffarian D, Rosenberg I, Uauy R. History of modern nutrition science—implications for current research, dietary guidelines, and food policy BMJ 2018; 361 :k2392 doi:10.1136/bmj.k2392 https://www.bmj.com/content/361/bmj.k2392 

    2. Williams, P. Nutritional composition of red meat. Nutr. Diet. 2007, 64, S113–S119. https://doi.org/10.1111/j.1747-0080.2007.00197.x
    https://onlinelibrary.wiley.com/doi/10.1111/j.1747-0080.2007.00197.x 
''')


if __name__ == '__main__':
    # run me with "streamlit run ..../MAHAT_PROJECT/app.py"
    main()