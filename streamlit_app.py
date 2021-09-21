from logging import warning
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import unicodedata
import requests

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

APP_NAME = 'Product Review Analysis'

HUGGINGFACE_API_TOKEN = ''
HUGGINGFACE_PARAPHRASE_API_URL = "https://api-inference.huggingface.co/models/tuner007/pegasus_paraphrase"

DISPLAY_MAX_REVIEWS = 100
MUST_INCLUDES = ['love', 'like', 'hate', 'cheap']
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

LABEL_PRODUCT_INFO = 'Product Information'
LABEL_TOP_NGRAMS = 'Top Ngrams'
LABEL_VIEW_REVIEWS = 'View Customer Reviews'
LABEL_AUTO_GENERATE_REVIEWS = 'Auto Generate Reviews'

def init_session():
    if 'selected_reviews' not in st.session_state:
        st.session_state.selected_reviews = {REVIEW_TYPE_HAPPY: [], REVIEW_TYPE_UNHAPPY: []}
def delete_session():
    for key in st.session_state.keys():
        del st.session_state[key]

def reset_session():
    delete_session()
    init_session()

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
        if (intersection >= len(words) and len(words) > 0) or (intersection >= len(words) - 1 and len(words) > 1):
            sentences.append(sen)
    return sentences

def download_text_file(label, data, file_name):
    st.download_button(label = label, data = data, file_name = file_name, mime='text/plain')

def view_settings():
    global MUST_INCLUDES, ADDITIONAL_STOPWORDS, DISPLAY_MAX_REVIEWS, TOTAL_NGRAMS, HAPPY_NGRAMS_TOTAL, UNHAPPY_NGRAMS_TOTAL, HUGGINGFACE_API_TOKEN, HUGGINGFACE_PARAPHRASE_API_URL
    st.sidebar.subheader('Settings')
    with st.sidebar.expander('General', False):
        MUST_INCLUDES = [w.strip() for w in st.text_input('Must Include Words', ', '.join(MUST_INCLUDES)).split(',')]
        DISPLAY_MAX_REVIEWS = int(st.number_input('Display Max Reviews', value=DISPLAY_MAX_REVIEWS))
    
    with st.sidebar.expander('Ngrams', False):
        ADDITIONAL_STOPWORDS = [w.strip() for w in st.text_input('Stopwords', ', '.join(ADDITIONAL_STOPWORDS)).split(',')]
        TOTAL_NGRAMS = int(st.number_input('Total Ngrams', 1, 50, TOTAL_NGRAMS))
        HAPPY_NGRAMS_TOTAL = int(st.number_input('Happy Ngram Length', 1, 5, HAPPY_NGRAMS_TOTAL))
        UNHAPPY_NGRAMS_TOTAL = int(st.number_input('Unhappy Ngram Length', 1, 5, UNHAPPY_NGRAMS_TOTAL))

    with st.sidebar.expander('Huggingface API', False):
        HUGGINGFACE_API_TOKEN = st.text_input('API Token', HUGGINGFACE_API_TOKEN)
        HUGGINGFACE_PARAPHRASE_API_URL = st.text_input('Paraphrase API URL', HUGGINGFACE_PARAPHRASE_API_URL)

def get_product_title(df):
    return df.iloc[0][COLUMN_PRODUCT_TITLE]

def get_happy_reviews_df(df):
    return df[df[COLUMN_RATING] >= 3]

def get_unhappy_reviews_df(df):
    return df[df[COLUMN_RATING] < 3]

def view_product_info(df):
    with st.expander(LABEL_PRODUCT_INFO, True):
        st.info(get_product_title(df))

        st.metric(label="Total Reviews", value=str(len(df)))
        st.bar_chart(data=df[COLUMN_RATING].value_counts())

def generate_top_ngrams(df):
    with st.expander(LABEL_TOP_NGRAMS):
        product_title_words = [w for w in basic_clean(get_product_title(df)) if len(w) > 2]
        if len(ADDITIONAL_STOPWORDS):
            product_title_words.extend(ADDITIONAL_STOPWORDS)
        product_stopwords = st.multiselect('Select Stopwords', sorted(product_title_words))

        happy_words = get_clean_words(get_happy_reviews_df(df)[COLUMN_REVIEW_TEXT], product_stopwords)
        unhappy_words = get_clean_words(get_unhappy_reviews_df(df)[COLUMN_REVIEW_TEXT], product_stopwords)

        happy_ngrams = get_ngrams(happy_words, HAPPY_NGRAMS_TOTAL)[:TOTAL_NGRAMS]
        unhappy_ngrams = get_ngrams(unhappy_words, UNHAPPY_NGRAMS_TOTAL)[:TOTAL_NGRAMS]

        display_ngram_chart(REVIEW_TYPE_HAPPY, happy_ngrams)
        display_ngram_chart(REVIEW_TYPE_UNHAPPY, unhappy_ngrams)

        return happy_ngrams, unhappy_ngrams

def view_reviews(happy_df, unhappy_df, happy_ngrams, unhappy_ngrams, selected_product_asin):
    global HUGGINGFACE_API_TOKEN, HUGGINGFACE_PARAPHRASE_API_URL
    st.title('Reviews')
    review_type = st.radio('Select Review Type', [REVIEW_TYPE_HAPPY, REVIEW_TYPE_UNHAPPY])
    
    ngram_list = happy_ngrams if review_type == REVIEW_TYPE_HAPPY else unhappy_ngrams
    selected_ngram = st.selectbox('Select Ngrams', merge_ngram_words(ngram_list))
    ngram_regex = ''.join(f'.*({w})' for w in selected_ngram.split(' '))

    selected_must_includes = st.multiselect('Select Must Include Words', MUST_INCLUDES)
    must_include_regex = '|'.join(r'\b(' + w + r')\b' for w in selected_must_includes)

    df = happy_df if review_type == REVIEW_TYPE_HAPPY else unhappy_df
    df = df[df[COLUMN_REVIEW_TEXT].str.contains(ngram_regex, regex=True, case=False, na=False)]
    if len(must_include_regex):
        df = df[df[COLUMN_REVIEW_TEXT].str.contains(must_include_regex, regex=True, case=False, na=False)]

    with st.expander(f'{review_type} Reviews ({len(df)})'):
        i = 0
        for original_review in df[COLUMN_REVIEW_TEXT][:DISPLAY_MAX_REVIEWS]:
            st.subheader(df.iloc[i][COLUMN_REVIEW_TITLE] + ' ' + '⭐' * int(df.iloc[i][COLUMN_RATING]))
            
            review = original_review
            ngram_words = list(set(selected_ngram.split(' ')))
            
            highlighted_wordss = list(set(ngram_words + selected_must_includes))
            for word in highlighted_wordss:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlight_color = '#aaffaa' if word in selected_must_includes else 'yellow'
                review = pattern.sub(f'<span style="background:{highlight_color}">' + word + '</span>', review)
            st.markdown(review, unsafe_allow_html=True)

            extracted_sents = list(set(extract_sentence(original_review, ngram_words) + extract_sentence(original_review, selected_must_includes)))
            if len(extracted_sents):
                j = 0
                for sent in extracted_sents:
                    container = st.container()
                    container.info(sent)

                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        should_select_review = True if sent.strip() in st.session_state.selected_reviews[review_type] else False
                        if st.checkbox('Select review', key='-'.join([review_type, selected_ngram, 'select-review-checkbox', str(i), str(j)]), value=should_select_review):
                            if sent.strip() not in st.session_state.selected_reviews[review_type]:
                                st.session_state.selected_reviews[review_type].append(sent.strip())
                        elif sent.strip() in st.session_state.selected_reviews[review_type]:
                            st.session_state.selected_reviews[review_type].remove(sent.strip())

                    with col2:
                        if len(HUGGINGFACE_API_TOKEN) and len(HUGGINGFACE_PARAPHRASE_API_URL):
                            if st.checkbox('Paraphrase', key='-'.join([review_type, selected_ngram, 'paraphrase-checkbox', str(i), str(j)])):
                                para_text = paraphrase(sent, HUGGINGFACE_API_TOKEN)
                                if len(para_text):
                                    container.warning(para_text)
                                    with col3:
                                        should_select_paraphrase = True if para_text.strip() in st.session_state.selected_reviews[review_type] else False
                                        if st.checkbox('Select paraphrase', key='-'.join([review_type, selected_ngram, 'select-paraphrase-checkbox', str(i), str(j)]), value=should_select_paraphrase):
                                            if para_text.strip() not in st.session_state.selected_reviews[review_type]:
                                                st.session_state.selected_reviews[review_type].append(para_text.strip())
                                        elif para_text.strip() in st.session_state.selected_reviews[review_type]:
                                            st.session_state.selected_reviews[review_type].remove(para_text.strip())
                                else:
                                    container.error('Unable to paraphrase this sentence')
                    j += 1
            else:
                st.error('Unable to extract sentence')

            st.write('---')
            i += 1
        st.markdown(f'*{min(DISPLAY_MAX_REVIEWS, len(df))} of {len(df)} reviews*')

    for review_type in [REVIEW_TYPE_HAPPY, REVIEW_TYPE_UNHAPPY]:                    
        total_selected_reviews = len(st.session_state.selected_reviews[review_type])
        if total_selected_reviews:
            with st.expander(f'Selected {review_type} Reviews ({total_selected_reviews})'):
                for review in st.session_state.selected_reviews[review_type]:
                    st.markdown('* ' + review)
                    st.write('')
                file_name = f'{selected_product_asin}-{review_type.lower()}-reviews.txt'
                data = review_type + ' Reviews:\n- ' + '\n- '.join(st.session_state.selected_reviews[review_type])
                download_text_file("Download", data, file_name)

def auto_generate_reviews(happy_df, unhappy_df, happy_ngrams, unhappy_ngrams):
    with st.expander(LABEL_AUTO_GENERATE_REVIEWS, True):
        pass

def main():
    st.set_page_config(APP_NAME, initial_sidebar_state='collapsed')
    st.title(APP_NAME)

    init_session()
    view_settings()

    data_file = st.file_uploader('Upload CSV', 'csv')
    if data_file:
        df = pd.read_csv(data_file)

        if len(df) == 0:
            st.warning('No products found')
        else:
            product_asin_list = list(df[COLUMN_ASIN].unique())

            selected_product_asin = st.selectbox('Select Product ASIN', product_asin_list, on_change=reset_session)
            
            df = df[df[COLUMN_ASIN] == selected_product_asin]

            if len(df) == 0:
                st.warning('No reviews found')
            else:
                view_product_info(df)

                happy_df = get_happy_reviews_df(df)
                unhappy_df = get_unhappy_reviews_df(df)

                happy_ngrams, unhappy_ngrams = generate_top_ngrams(df)

                should_display_reviews = st.checkbox(LABEL_VIEW_REVIEWS)
                should_auto_generate_reviews = st.checkbox(LABEL_AUTO_GENERATE_REVIEWS)                

                if should_display_reviews:
                    view_reviews(happy_df, unhappy_df, happy_ngrams, unhappy_ngrams, selected_product_asin)
                                    
                if should_auto_generate_reviews:
                    auto_generate_reviews(happy_df, unhappy_df, happy_ngrams, unhappy_ngrams)
main()