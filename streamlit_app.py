from logging import warning
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import unicodedata
import requests

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

APP_NAME = 'Product Review Analysis'

HUGGINGFACE_PARAPHRASE_API_URL = "https://api-inference.huggingface.co/models/tuner007/pegasus_paraphrase"

DISPLAY_MAX_REVIEWS = 100
ADDITIONAL_STOPWORDS = []
TOTAL_NGRAMS = 20
HAPPY_NGRAMS_TOTAL = 3
UNHAPPY_NGRAMS_TOTAL = 2

COLUMN_ASIN = 'asin'
COLUMN_PRODUCT_TITLE = 'product_title'
COLUMN_RATING = 'rating'
COLUMN_REVIEW_TITLE = 'review_title'
COLUMN_REVIEW_TEXT = 'review_text'

REVIEW_TYPE_HAPPY = 'Happy'
REVIEW_TYPE_UNHAPPY = 'Unhappy'

def basic_clean(text, product_stopwords = []):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + product_stopwords
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

def get_clean_words(review_column, product_stopwords):
    return basic_clean(''.join(str(review_column.tolist())), product_stopwords)

def get_ngrams(words, ngram=2):
    return (pd.Series(nltk.ngrams(words, ngram)).value_counts())

def merge_ngram_words(ngram_series):
    return [' '.join(words) for words in ngram_series.index.tolist()]

def display_ngram_chart(title, ngram_series):
    fig, ax = plt.subplots()
    ax.set_xlabel(title + ' Ngrams')
    ax.barh(merge_ngram_words(ngram_series), ngram_series.values.tolist())
    ax.invert_yaxis()
    st.pyplot(fig)

def get_auth_header(token):
    return {"Authorization": f"Bearer {token}"}

@st.cache
def paraphrase(text, token):
    response = requests.post(HUGGINGFACE_PARAPHRASE_API_URL, headers=get_auth_header(token), json={"inputs": text})
    if response and len(response.json()):
        return response.json()[0]['generated_text']
    return ''

def extract_sentence(text, words):
    sentences = []
    for sen in sent_tokenize(text):
        l = word_tokenize(sen)
        intersection = len(set(l).intersection(words))
        if intersection >= len(words) - 1 and len(words) > 1:
            sentences.append(sen)
    return sentences

def main():
    global ADDITIONAL_STOPWORDS, DISPLAY_MAX_REVIEWS, HUGGINGFACE_PARAPHRASE_API_URL, TOTAL_NGRAMS, HAPPY_NGRAMS_TOTAL, UNHAPPY_NGRAMS_TOTAL

    huggingface_api_token = ''

    st.set_page_config(APP_NAME, initial_sidebar_state='collapsed')
    st.title(APP_NAME)

    st.sidebar.subheader('Settings')
    with st.sidebar.expander('General', False):
        DISPLAY_MAX_REVIEWS = int(st.number_input('Display Max Reviews', value=DISPLAY_MAX_REVIEWS))
    
    with st.sidebar.expander('Ngrams', False):
        ADDITIONAL_STOPWORDS = [w.strip() for w in st.text_input('Stopwords (comma seperated)', ', '.join(ADDITIONAL_STOPWORDS)).split(',')]
        TOTAL_NGRAMS = int(st.number_input('Total Ngrams', 1, 50, TOTAL_NGRAMS))
        HAPPY_NGRAMS_TOTAL = int(st.number_input('Happy Ngram Length', 1, 5, HAPPY_NGRAMS_TOTAL))
        UNHAPPY_NGRAMS_TOTAL = int(st.number_input('Unhappy Ngram Length', 1, 5, UNHAPPY_NGRAMS_TOTAL))

    with st.sidebar.expander('Huggingface API', False):
        huggingface_api_token = st.text_input('API Token', huggingface_api_token)
        HUGGINGFACE_PARAPHRASE_API_URL = st.text_input('Paraphrase API URL', HUGGINGFACE_PARAPHRASE_API_URL)

    data_file = st.file_uploader('Upload CSV', 'csv')
    if data_file:
        df = pd.read_csv(data_file)

        if len(df) == 0:
            st.warning('No products found')
        else:
            product_asin_list = list(df[COLUMN_ASIN].unique())

            selected_product_asin = st.selectbox('Select Product ASIN', product_asin_list)
            
            df = df[df[COLUMN_ASIN] == selected_product_asin]

            if len(df) == 0:
                st.warning('No reviews found')
            else:
                product_title = df.iloc[0][COLUMN_PRODUCT_TITLE]
                st.info(product_title)

                st.metric(label="Total Reviews", value=str(len(df)))
                st.bar_chart(data=df[COLUMN_RATING].value_counts())

                happy_reviews = df[df[COLUMN_RATING] >= 3]
                unhappy_reviews = df[df[COLUMN_RATING] < 3]

                st.header('Top Ngrams')

                product_title_words = [w for w in basic_clean(product_title) if len(w) > 2]
                if len(ADDITIONAL_STOPWORDS):
                    product_title_words.extend(ADDITIONAL_STOPWORDS)
                product_stopwords = st.multiselect('Select Stopwords', sorted(product_title_words))

                happy_words = get_clean_words(happy_reviews[COLUMN_REVIEW_TEXT], product_stopwords)
                unhappy_words = get_clean_words(unhappy_reviews[COLUMN_REVIEW_TEXT], product_stopwords)

                happy_ngrams = get_ngrams(happy_words, HAPPY_NGRAMS_TOTAL)[:TOTAL_NGRAMS]
                unhappy_ngrams = get_ngrams(unhappy_words, UNHAPPY_NGRAMS_TOTAL)[:TOTAL_NGRAMS]

                review_type = st.radio('Select Review Type', [REVIEW_TYPE_HAPPY, REVIEW_TYPE_UNHAPPY])

                ngram_list = happy_ngrams if review_type == REVIEW_TYPE_HAPPY else unhappy_ngrams

                display_ngram_chart(review_type, ngram_list)
                

                st.header('Reviews')
                selected_reviews = []

                selected_ngram = st.selectbox('Select Ngrams', merge_ngram_words(ngram_list))

                df = happy_reviews if review_type == REVIEW_TYPE_HAPPY else unhappy_reviews
                filtered_df = df[df[COLUMN_REVIEW_TEXT].str.contains(''.join(f'.*({w})' for w in selected_ngram.split(' ')), regex=True, case=False, na=False)]
                
                with st.expander(f'{review_type} Reviews ({len(filtered_df)})'):
                    i = 0
                    for original_review in filtered_df[COLUMN_REVIEW_TEXT][:DISPLAY_MAX_REVIEWS]:
                        st.subheader(filtered_df.iloc[i][COLUMN_REVIEW_TITLE] + ' ' + '⭐' * int(filtered_df.iloc[i][COLUMN_RATING]))
                        
                        review = original_review
                        words = list(set(selected_ngram.split(' ')))
                        
                        for word in words:
                            pattern = re.compile(re.escape(word), re.IGNORECASE)
                            review = pattern.sub('<span style="background:yellow">' + word + '</span>', review)
                        st.markdown(review, unsafe_allow_html=True)


                        extracted_sents = extract_sentence(original_review, words)
                        if len(extracted_sents):
                            j = 0
                            for sent in extracted_sents:
                                container = st.container()
                                container.info(sent)

                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if st.checkbox('Select review', key='select-button-' + str(i) + str(j)):
                                        selected_reviews.append(sent.strip())

                                with col2:
                                    if len(huggingface_api_token) and len(HUGGINGFACE_PARAPHRASE_API_URL):
                                        if st.checkbox('Paraphrase', key='paraphrase-button-' + str(i) + '-' + str(j)):
                                            para_text = paraphrase(review, huggingface_api_token)
                                            if len(para_text):
                                                container.warning(para_text)
                                                with col3:
                                                    if st.checkbox('Select paraphrase', key='select-paraphrase-' + str(i) + '-' + str(j)):
                                                        selected_reviews.append(para_text.strip())
                                            else:
                                                container.error('Unable to paraphrase this sentence')
                                j += 1
                        else:
                            st.error('Unable to extract sentence')

                        st.write('---')
                        i += 1
                    st.markdown(f'*{min(DISPLAY_MAX_REVIEWS, len(filtered_df))} of {len(filtered_df)} reviews*')
                
                with st.expander(f'Selected {review_type} Reviews ({len(selected_reviews)})'):
                    for review in selected_reviews:
                        st.markdown('* ' + review)
                        st.write('')
                    if len(selected_reviews):
                        st.download_button(
                            label = f"Download",
                            data=review_type + ' Reviews:\n- ' + '\n- '.join(selected_reviews),
                            file_name = f'{selected_product_asin}-{review_type.lower()}-reviews.txt',
                            mime='text/plain'
                        )

main()