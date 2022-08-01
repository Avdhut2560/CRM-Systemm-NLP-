# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 20:20:23 2022

@author: avdhu
"""

#topic modeling POC
import streamlit as st
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.stem import SnowballStemmer
import re
import string
import pickle


cv = pickle.load(open("cv_vectorizer.pkl", "rb"))
lda_model = pickle.load(open("lda_model_new.pkl", "rb"))
stop_words2 = pickle.load(open("stopwords.pkl","rb"))
tf = pickle.load(open("tf_vectorizer_tp.pkl","rb"))
tf2 = pickle.load(open("tf_vecorizer_cm.pkl","rb"))
model = pickle.load(open("GNB_new.pkl","rb"))
pun_word = string.punctuation
lamma = SnowballStemmer("english")
# lets try lemmatization
def preprocess2(txt):
    x = txt.lower()
    x = re.sub("\d+[/?]\w+[/?]\w+:|\d+[|]\w+[|]\w+:|\d+[/]\w+[/]\w+[(]\w+[)]:?", "", x)
    x = re.sub("int[a-z]+d$", "interested", x)
    x = re.sub("[\d+-?,'.]", "", x)
    x = [i for i in nltk.word_tokenize(x) if i not in stop_words2 and len(i)>1 and i not in pun_word] # word_tokenization, stop_word/punctuation removal
    x = [lamma.stem(i) for i in x] # there are still a lot of incorrect spellings
    return " ".join(x)

# Classification System
def Status(user,loc,execu):
    x = user
    y = loc.lower()
    z = execu.lower()
    x = preprocess2(x)
    x = y +" "+ z +" "+ x
    x = tf2.transform([x])
    x = model.predict(x.toarray())
    if x == 1:
        return "Not Convertable"
    else:
        return  "Convertable"
    
# Topic Modeling System  
def topic_model(user):
    user_msg = user
    x = preprocess2(user_msg)
    x = tf.transform([x])
    lda_x = lda_model.transform(x)
    tpic = []
    tpc = lambda x : "Not Interested" if x == 0 else "Interested"
    for i,topic in enumerate(lda_x[0]):
        tpc_name = tpc(i)
        prc = np.round(topic*100,2)
        tpic.append([tpc_name,prc])
    return tpic
    

    
df = pd.DataFrame({"Location" : ["bangalore","hyderabad"],
         "Buisness Executive" : ["prema","surendra"],
         "Basic Conversation ": ["""24/7/prema: RNR 25/7/prema: rnr 29/7/prema: nr 1/8/prema: rnr ; sms sent 4/8/prema: inquired regarding the placements and live internship
                                 """,
                                 """8/6/17(Surendra):call me after some time iam busy now 12/6/17(Surendra):share me details i will check evening session 13/6/17(Surendra):call me after iam busy 1/8/17(Surendra):i will come share me details"""]},
                  index = ['Example1','Example2'])

    
    
def main():
    #Adding title
    st.markdown('<style>body{background-color: #021c1e;}</style>',unsafe_allow_html=True)
    st.markdown("""<div style="background-color:{};padding:10px;border-radius:18px">
    <h1 style="color:white;text-align:center;">CRM Classification System</h1>
    </div>
    """.format('#86ac41'),unsafe_allow_html=True)
    
    st.markdown("***")

    #Adding sentence or just writing article
    st.markdown("""<div style="background-color:{};padding:10px;border-radius:0px">
    <h1 style="font-size:15px;
    color:white;
    text-align:center;
    font-family:Courier;">
    It provides details about whether the person is "Converted" or "Not Converted" based on location, business executive, 
    and basic conversation between client and business executive.
    </h1>
    </div>
    """.format('#1e656d'),unsafe_allow_html=True)
    
    st.markdown("")
    
    with st.expander("Examples"):
        st.table(df)
    #Adding selectbox for the location
    loc = st.selectbox("Choose a location :",["None",'ahmedabad', 'aurangabad', 'australia', 'bangalore', 'bihar',
       'bilgi', 'chennai', 'coimbatore', 'delhi', 'faridabad', 'ghazibad',
       'gujarat', 'guntur', 'gurgaon', 'hubli', 'hyderabad', 'india',
       'jaipur', 'jalandhar', 'kadapa', 'kerala', 'khammam', 'kochi',
       'kolkatta', 'madurai', 'meerut', 'mumbai', 'mysore', 'nagpur',
       'nasik', 'nepal', 'noida', 'ongole', 'pune', 'rajamundry',
       'rayagada', 'solapur', 'thane', 'tiruttani', 'uae', 'usa',
       'vijayawada', 'vishakapatnam'])
    
    st.markdown("")

    #Adding selectbox for the business executive 
    execu = st.radio("Select a business executive :",['None', 'prema', 'mohan', 'surendra', 'soma', 'amar', 'sankar', 'gowtham',
       'sai'])
    
    st.markdown("") 
    #Adding text box for text 
    user = st.text_area("Paste your chats/mgs here for topic modeling :")
    
    #following lines of code for prediction
    if st.button("Predict"):
        result =Status(user,loc,execu)
        if result == "Convertable":
            st.success(result)
        else:
            st.error(result)
            

        
        
    st.markdown("***") 
    #Adding sentence or just writing article
    st.markdown("""<div style="background-color:{};padding:10px;border-radius:0px">
    <h1 style="font-size:15px;
    color:#021C1E;
    text-align:center;
    font-family:Cambria;">
    It provides details about whether the customer shows interest
   or not towards our product based on the above conversation.
    </h1>
    </div>
    """.format('#6fb98f'),unsafe_allow_html=True)#f0810f #6fb98f
    
    st.markdown(" ") 
    st.markdown(" ")
    #Follwing lines of code show whether the customer or client interested or not in product
    if st.button("Model the topic"):
        topics = topic_model(user)
        for i in topics:
            st.write(i)
    st.markdown("***")
    
    
    
    
   
    
    
    
    


    st.write("## Thank you for Visiting \nProject by Avdhut")
    st.markdown("<h1 style='text-align: right; color: #d7e3fc; font-size: small;'><a href='https://github.com/'>Other Works</a></h1>", unsafe_allow_html=True)
    
    


if __name__ == "__main__":
    main()
