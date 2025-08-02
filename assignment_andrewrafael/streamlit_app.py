import streamlit as st
import requests
import pandas as pd

st.title("News Headline Sentiment Analyzer")

if "user_headline" not in st.session_state:
    st.session_state.user_headline = ""  # Initialize session state variable if it doesn't exist

def clear_text():
    st.session_state.user_headline = ""  # Clear the text area by resetting the session state

headline = st.text_area("Enter a news headline:", 
                        key = 'user_headline', 
                        placeholder="e.g. 'Stock market hits record high, analysts optimistic'")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    analyze_sentiment_clicked = st.button("Analyze Sentiment")

with col2:
    st.button("Delete Text", on_click = clear_text)

st.feedback("thumbs")

HOST = 'localhost'
PORT =  8009

if analyze_sentiment_clicked:
    list_of_headlines = [headline.strip() for headline in headline.split(",")]
    if list_of_headlines:
        response = requests.post(url=f'http://{HOST}:{PORT}/score_headlines',
                         json={'headlines': list_of_headlines})
        if response.status_code == 200:
            my_result = []
            result = response.json()
            for idx, label in enumerate(result['labels']):
                my_result.append({'headline': list_of_headlines[idx],
                                  'sentiment': label})
            
            df_my_result = pd.DataFrame(my_result)

            st.dataframe(df_my_result)

            # st.success(f"Sentiment: {result['labels']}")
        else:
            st.error("Error analyzing sentiment.")
    else:
        st.warning("Please enter a headline to analyze.")