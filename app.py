import os
from re import X
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests
# from recommender_system import show_recommendations
from PIL import Image

__DIRNAME__ = os.path.dirname(os.path.realpath(__file__))
DATASET = pd.read_csv(os.path.join(__DIRNAME__, 'israeli_data.csv'))

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


def select_box(dicty):
    pic = st.selectbox("Title", list(dicty.keys()))
    st.image(dicty[pic], use_column_width=True)


# def recommend(item):
#   r1, r2, r3 = show_recommendations(item)
#   st.markdown(''' #####  המלצה לפי טקסט דומה ''')
#   for i, v in r1.iteritems():
#     v
#   st.markdown(''' #####  המלצה לפי ערכי מאקרו דומים ''')
#   for i, v in r2.iteritems():
#     v
#   st.markdown(''' ##### המלצה לפי ערכי מאקרו ומיקרו דומים ''')
#   for i, v in r3.iteritems():
#     v

def main():
    lottie_book = load_lottieurl('https://assets2.lottiefiles.com/temp/lf20_nXwOJj.json')
    st_lottie(lottie_book, speed=1, height=200, key="initial")
    one, row0_1, three, row0_2, two = st.columns((.1, 2, .2, 1, .1))

    row0_1.title('Food Group')

    with row0_2:
        st.write('')

    row0_2.subheader(
        'A Streamlit web app by Barel Mishal, Sapir Shapira, Yoav Orenbach and Hen Emuna')

    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

    with row1_1:
        st.header('**Motivation and problem description:**')
        st.markdown("Food & Nutrition Science is a newly developed field critical for health and development. It affects our body’s homeostasis in general, specifically our immune system, sugar balance, endocrine system, etc. As a young research field, there is a need to deepen the field’s knowledge further. Moreover, we cannot partially leverage existing knowledge, as it contains many myths and biases, some of which are incorrect. An excellent example of this bias is the relation to fat, which is incorrectly considered unhealthy. In fact, it is the vast intake of sugar that comes from this thinking that causes many of the diseases today. As a result, Nutritionists must use both reliable information and scientifically supported tools to know the exact composition of food. This knowledge can be used both for personal diet recommendation as well as interdisciplinary research that may link the different nutritions to metabolic processes and a wide range of non-communicable diseases: type 2 diabetes, some developmental issues, and many more [1].")
        st.markdown(
            "A critical aspect of daily nutritionist’s work is to suggest dietary alterations for patients, based on their specific health profile, using previously defined food groups (s.a., Meat, milk, vegetables, fruits, and sweets). The rationale behind the clustering to food groups,  based on known macronutrients (carbohydrates, fats, and proteins) and an estimation of the equivalent micronutrients (Vitamins, and minerals), is to enable the creation of reliable tools, which would provide nutritionists and their patients the ability to create dietary alternatives and control regarding their food consumption.")
        st.markdown(
            'Nowadays, this clustering relies mainly on assumptions and lists of products learned by heart by Nutritionists.  ')

    row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
    with row2_1:
        st.header('**Data:**')

        st.markdown("Israeli nutrition data source (https://data.gov.il/dataset/nutrition-database) from the ministry of health. A table (10MBs) with 4650 records of 74 nutritional components: protein, fats, carbohydrates, amino acids, fatty acids, vitamins, and minerals. The table was created from a JSON file containing records and metadata, which was pre-processed (units conversion, missing fields, dropping non-helpful features like psolet, Supplements, and Recipes) and saved as a CSV. In addition, we also use the U.S department of agriculture Food Data Central (FDC) dataset, which the Israeli data relies on. After using the FDC API key to extract information, we built a table of 7600 records of 149 nutritional components containing the components of the firs table.")
    
    row2_spacer1, row_3, row2_spacer2 = st.columns((.1, 3.2, .1))
    with row_3:
        st.header('**Solution:**')
        
        st.markdown("Our goal is to improve the existing assumptions and biases used in the Israeli nutrition community, thus helping nutritionists offer better alternatives for their patience. In particular, we intend to make improvements in three areas:")
        
        st.subheader('**1.** Clustering of food items to food groups.')
        st.markdown(
            '''
        There are no known food group labels for food items in the Israeli data (or the American data for that matter), since some food items cannot be classified into one food group, but to several. For example, if we look at a food item - English breakfast, it could fall into the meat food group as it has many meats, however, it could also fall into the fat food group because of the high amounts of fat in an English breakfast. Many more examples follow, though there are some food items which we unequivocally know their food group, like milk for instance.
        For this matter, we received a list from the ministry of health containing all the Israeli food groups (32) with known food items in each group, and our objective is to classify the different food items in the Israeli data into their respective food groups. We use the known ones and let the clustering algorithms answer the question of how to label food items that falls under more than one food group category.
        
        To better alabort for each algorithms we have created a dashboard and share it as heroku site - 
        
        *************  https://food-group.herokuapp.com/ *************
        
        We applied many clustering algorithms – Kmeans, Agglomerative, DBScan and mean-shift, and show the results in a Heroku dashboard site where one can switch between the algorithms and see a plot of the results, along with a mapping table between food groups and the clustering labels (detailed in the evaluation section)
        '''
        )

        st.subheader('**2.** Predicting food’s micronutrients based on their macronutrients using machine and deep learning principles.')
        
        st.markdown('''    
            Since the Israeli data is quite small, we decided to use the FDC data as part of our training data, and since we want to be successful with the Israeli data, our test set only contained food items from there.
            Our input contains the macronutrients – proteins, carbohydrates and fats, while our output contains the micronutrients – vitamins and minerals.
            
            We applied many machine learnings algorithms since the connection between macronutrients and micronutrients is not pre-established. Specifically we used:
                
                Linear regression
            
                K-nearest neighbors
            
                Decision tree regressor
            
                Random forest regressor
                
                Neural network (multi-layer perceptron)

                
        '''
        )

    row3_space1, row4_1, row3_space2, row4_2, row3_space3 = st.columns(
        (.1, 1, .1, 1, .1))

    with row4_1:
        path = os.path.join(__DIRNAME__, 'results', 'index.jpg')
        st.image(path, use_column_width=True)
    with row4_2:
        path2 = os.path.join(__DIRNAME__, 'results', 'results_results_multi_reg.png')
        st.image(path2, use_column_width=True)

    row2_spacer1, row_7, row2_spacer2 = st.columns((.1, 3.2, .1))
    with row_7:
        st.subheader('**3.** Recommendation of food alternatives based on similar products within the same food group as well as micronutrient estimation.')

        st.markdown(
            '''
            Recommendation of food alternatives based on similar products within the same food group as well as micronutrient estimation.
            
            Since there are no known ratings and we did not have enough time to gather ratings for the food items in our data, we decided to use content based recommender system. One of the downsides of using content based recommender is that finding the appropriate features for each food item profile is hard. Therefore, we used three different feature vectors:
            a)    Food items names
            b)    Food items macronutrients
            c)    Food items macronutrients + micronutrients
            As for the prediction heuristic, given a food item and any of the item’s profiles, we compute the cosine similarity between them and return the top ten most similar food items (based on that profile). We are left with thirty recommendations and filter any repeated ones.
        '''
        )

    line1_spacer1, line1_1, line1_spacer2 = st.columns((.1, 3.2, .1))

    with line1_1:



        st.header('Evaluation:')
        
    st.write('')
    row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
        (.1, 1, .1, 1, .1))

    with row3_1:
        st.subheader('1.    Evaluating food groups cluster:')

        st.markdown(
                '''                

    Seeing as there are no labels of food groups, we opted to use domain knowledge to evaluate our clustering. As noted before, we received a list from the ministry of health containing the known food items of each one of the 32 food groups and used this knowledge for our evaluation.
    First, we mapped 32 food items from the data into their respective food group. Then, after applying a clustering algorithm, we mapped the cluster labels of the 32 food items to the food groups we already assigned. For instance, if food item 1 was mapped to the meat food group and the clustering algorithm put it in the third cluster (so cluster label 3), than label 3 maps to the meat food group.
    If we manage to receive one-to-one mapping between the algorithm’s clustering labels to the food groups (so every food group has a single label), we can consider the clustering as a success. However, if a food group was given more than one label, than we know the clustering was not perfect, as we know from the ministry of health that the food items we used belong to different food groups.
                '''
                )
        select_word_cloud()
    
    with row3_2:
        st.subheader('2.    Evaluating micronutrients prediction:')

        st.markdown(
            '''
            The prediction task is a multi-output regression problem, thus the straightforward way to evaluate it is by splitting our data into train and test sets and computing our accuracy on the test set. However, we are dealing with numbers in the Micro-gram or Nano-gram ranges, so it was proven quite difficult to achieve high accuracy with exact numbers. Therefore, we wanted to know how far away we are from the exact micronutrient value. For this reason, when calculating the accuracy we decided to give a small error range of one micro-gram and we can consider our predictive model to be successful if he is correct with an error on one micro-gram.
            '''
            )

    st.write('')
    row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
        (.1, 1, .1, 1, .1))

    with row4_1:
        st.subheader('3.    Evaluating food alternatives recommendation:')
        recommender()

        # st.pyplot(fig)
    
        st.markdown(
            '''
            Due to the fact there are no ratings of food items we can use, we cannot evaluate our recommender system by creating a test set and computing the root-mean-square error for example. However, there is a way for us to know if our recommender is indeed successful and that is by using precision and recall.
    Let’s remember that:

    (formula)

    Given a food item, for each feature vector we recommend 5 items which we consider as positives, and the next 5 items we would have recommended we consider as negatives (we can’t treat all the foods in the table as negative, since most of them have nothing to do with a single food item, so we would wrongly get high recall).
    In order to calculate precision and recall we use our first step – the food groups clustering. Since we want to offer alternative food items within the same food group, we compare to food group label of the given food item to the food group label of the recommended items. For the 5 positive recommendations, if the labels match than we consider it a true positive, otherwise it is a false positive. Similarly, for the 5 negative recommendations, if the labels match than we consider it a false negative (a food item we should have recommended), otherwise it is a true negative. Afterwards, we compare the different feature vectors of the recommendations on different food items to see which has the highest precision and recall.
            '''
        )

    with row4_2:
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