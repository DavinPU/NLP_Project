#!/usr/bin/env python
# coding: utf-8

# In[14]:


from Sentiment_Analysis import getResults
import streamlit as st
import pandas as pd
import numpy as np

@st.cache
def fetch_data(topic):
    # cache this function as getResults() takes the longest to run
    results = getResults(topic)

    presidents_unique = results['president'].unique()
    earliest_quote = min(results['year'])
    latest_quote   = max(results['year'])

    return results, presidents_unique, earliest_quote, latest_quote



st.title("President Speech Lookup App")
st.write("""
Input a topic below and find snippets of president speeches and debates that discuss this topic! The results
will be ordered based on sentiment (positive/negative)""")

form = st.form(key="my_form")
topic = form.text_input(label="Enter Topic")
submit = form.form_submit_button(label="Analyze")
filter_results = False
if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

# get value of n
n = st.sidebar.number_input(label='No. of results per sentiment: ', min_value=1, max_value=8, value=3)

if filter_results:
    st.header("Results")
elif submit or st.session_state['submitted']:
    # store in session state so the script doesn't run submit again
    st.session_state['submitted'] = True

    # run the function to get the results
    results, presidents_unique, earliest_quote, latest_quote = fetch_data(topic)

    president_selectbox = st.sidebar.multiselect(
        'Filter Presidents',
        options=presidents_unique,
        default=presidents_unique
    )
    # filter out presidents based on selectbox
    results = results[results['president'].isin(president_selectbox)]

    # filter on positive/negative sentiment via checkbox
    positive_check = st.sidebar.checkbox("Positive Statements", value=True)
    negative_check = st.sidebar.checkbox("Negative Statements", value=True)

    # filter on time range via slider
    slider_range = st.sidebar.slider("Time Range",
                                     earliest_quote, latest_quote, (earliest_quote, latest_quote))

    results = results[results['year'] >= slider_range[0]]
    results = results[results['year'] <= slider_range[1]]


    # print out the results
    st.header("Results")

    # print out positive results
    if positive_check:
        positive_results = results.nlargest(n, 'score')
        st.markdown('<p style="color:LawnGreen; font-size: 24px;"> Positive Results </p>', unsafe_allow_html=True)
        for i in range(len(positive_results)):
            quote = str(i + 1) + '. "' + positive_results.iloc[i]['sentence'] + '" - ' + positive_results.iloc[i]['president'] + \
            ', ' + str(positive_results.iloc[i]['year']) + ' (score: ' + str(positive_results.iloc[i]['score'].round(2)) + ')'
            st.write(quote)

    # print out negative results
    if negative_check:
        negative_results = results.nsmallest(n, 'score')
        st.markdown('<p style="color:Red; font-size: 24px;"> Negative Results </p>', unsafe_allow_html=True)
        for i in range(len(negative_results)):
            quote = str(i + 1) + '. "' + negative_results.iloc[i]['sentence'] + '" - ' + negative_results.iloc[i]['president'] + \
            ', ' + str(negative_results.iloc[i]['year']) + ' (score: ' + str(negative_results.iloc[i]['score'].round(2)) + ')'
            st.write(quote)

