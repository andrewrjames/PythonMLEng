import streamlit as st
import requests
import pandas as pd

st.title("News Headline Sentiment Analyzer")

headline = st.text_area("Enter a news headline:")

st.feedback("thumbs")

HOST = 'localhost'
PORT = 8009


if st.button("Analyze Sentiment"):
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