import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud
import streamlit as st
from IPython import get_ipython
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

fake = pd.read_csv("data\Fake.csv")
true = pd.read_csv("data\True.csv")

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

# UI:
st.sidebar.header("Navigation")
rad = st.sidebar.radio(
    "Menu ", ["Home", "News Verification", "Dataset", "About us"]
)

if rad == "Home":
    st.title("Insight News")
    st.caption("Verify your News & Eradicate Fake News")
    st.image("images\image0.webp", caption="The future is Here", use_column_width=True) 

def get_weight_matrix(model):
        weight_matrix = np.zeros((vocab_size, DIM))

        for word,i in vocab.items():
            weight_matrix[i] = model.wv[word]

        return weight_matrix
result = 1

if rad == "News Verification":
    st.title("News Verification")
    st.markdown(
        """ 
    ### Welcome to Insight News! <br>

    """,
        True,
    )
    st.write(
        """
            ## **ENTER YOUR NEWS TO BE VERIFIED**
            # """
    )
    user_input = st.text_input("Input")
    u_submit = st.button("VERIFY")
    
    if u_submit:
    
        if user_input !="":
            
            if result == 1:                                     
                st.markdown("This is a Real News 👍")
                st.markdown("Your News: ")
                st.write(user_input) 
            else:
                st.markdown("This is a Fake News 👎")
                st.write("Your News: ")
                st.write(user_input)
                                
if rad == "Dataset":
    st.title("Dataset")              
    
    st.header('True News')
    st.image('images/trueh.png')
    st.image("images/true.png", use_column_width=True)
    #st.table(true.iloc[0:10])
    #text = ' '.join(true['text'].tolist())
    #wordcloud = WordCloud().generate(text)
    #plt.imshow(wordcloud)
    
    st.header('False News')
    st.image("images/fakeh.PNG")
    st.image("images/fake.PNG", use_column_width=True)
    #fake.head()
    #text = ' '.join(fake['text'].tolist())
    #wordcloud = WordCloud().generate(text)
    #plt.imshow(wordcloud)

    

if rad == "About us":
    st.title("About us")
    st.caption("")
    st.markdown(
    """ 
    # Insight News - Verify your News & Eradicate Fake News: 

Lots of things you read online especially in your social media feeds may appear to be true, often are not. 
Fake news is news, stories or hoaxes created to deliberately misinform or deceive readers. 
Usually, these stories are created to either influence people’s views, push a political agenda or cause confusion and can often be a profitable business for online publishers. 
Fake news stories can deceive people by looking like trusted websites or using similar names and web addresses to reputable news organisations. 
Many people now get news from social media sites and networks and often it can be difficult to tell whether stories are credible or not. 
Information overload and a general lack of understanding about how the internet works by people has also contributed to an increase in fake news or hoax stories. Social media sites can play a big part in increasing the reach of these types of stories. **
Our aim is to implement a model which classifies news articles as fake or legitimate. 
The user can enter the headline for any news and the system classifies it, giving out the result as the news being legitimate or fake.

    """,
        True,
    )
       