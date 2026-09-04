import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download NLTK datasets safely
@st.cache_resource
def load_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

load_nltk()

# 1. FAQ Dataset
faq_db = [
    {
        "question": "What is your return policy?",
        "answer": "Our return policy allows you to return products within 30 days of purchase. Please ensure that the items are in their original condition and packaging."
    },
    {
        "question": "What is this AI chatbot designed for?",
        "answer": "This chatbot answers Frequently Asked Questions (FAQs) using NLP techniques like TF-IDF and Cosine Similarity."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can contact our customer support team via email at support@company.com."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept various payment methods including credit cards, debit cards, paypal, and bank transfers."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Shipping times vary based on your location. Typically, it takes 5-7 business days from the date of order confirmation."
    },
    {
        "question": "Which programming libraries and language are used?",
        "answer": "The system is built with python using NLTK for preprocessing, scikit learn for similarity matching, and streamlit for the user interface."
    },
    {
        "question": "How does text preprocessing work?",
        "answer": "Preprocessing converts text to lowercase, removes punctuation/stopwords, and tokenizes to words to clean input queries."
    },
    {
        "question": "What is the purpose of this chatbot?",
        "answer": "The purpose of this chatbot is to provide quick and accurate responses to frequently asked questions about the company's products and services."
    },
    {
        "question": "How can I provide feedback on the chatbot?",
        "answer": "You can provide feedback about the chatbot by contacting our support team at support@company.com."
    }
]

questions_list = [faq["question"] for faq in faq_db]
answers_list = [faq["answer"] for faq in faq_db]

# 2. Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

preprocessed_questions = [preprocess_text(question) for question in questions_list]

# 3. Match User Query with Cosine Similarity
def get_best_response(user_query: str, threshold: float = 0.2) -> str:
    cleaned_user_query = preprocess_text(user_query)
    
    if not cleaned_user_query.strip():
        return "Please ask a valid question with words."

    corpus = preprocessed_questions + [cleaned_user_query]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    best_match_idx = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_idx]

    if best_score < threshold:
        return "I'm sorry, I couldn't find a confident match for your question. Could you please rephrase?"

    return answers_list[best_match_idx]

# 4. Streamlit UI
st.set_page_config(page_title="FAQ AI Chatbot", page_icon="🤖")
st.title("🤖 Intelligent FAQ Chatbot")
st.caption("Ask any question related to our system and NLP technology.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    bot_response = get_best_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.chat_message("assistant").write(bot_response)