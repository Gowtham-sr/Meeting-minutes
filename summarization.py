from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from sklearn.model_selection import GridSearchCV
import re
from collections import Counter
from rouge_score import rouge_scorer

def summarize_with_tfidf(text, num_sentences=10):
    sentences = text.split('. ')
    sentences = [s.strip() for s in sentences if s]

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores, axis=0)[-num_sentences:]]
    
    summary =  '. '.join(ranked_sentences)
    print("TF-IDF ROUGE Score:")
    print(calculate_rouge_score(summary, text))
    return summary

def summarize_with_lda(text, num_topics=5, num_sentences=10):
    sentences = text.split('. ')
    sentences = [s.strip() for s in sentences if s]

    vectorizer = CountVectorizer(stop_words='english', max_df=0.85)
    X = vectorizer.fit_transform(sentences)

    lda = LatentDirichletAllocation(random_state=0)
    lda_params = {'n_components': [5, 10, 15], 'max_iter': [10, 20, 30]}
    lda_grid_search = GridSearchCV(lda, lda_params, cv=3)
    lda_grid_search.fit(X)

    best_lda = lda_grid_search.best_estimator_
    topic_distribution = best_lda.transform(X)
    sentence_scores = np.max(topic_distribution, axis=1)

    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores, axis=0)[-num_sentences:]]
    summary ='. '.join(ranked_sentences)
    print(calculate_rouge_score(summary, text))
    return summary

def advanced_summarize_text(text):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=1024,  
        truncation=True
    )
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=500,
        min_length=200,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(calculate_rouge_score(summary, text))
    return summary

def summarize_with_frequency(text, num_sentences=10):
    sentences = text.split('. ')
    sentences = [s.strip() for s in sentences if s]
    
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)
    
    sentence_scores = []
    for sentence in sentences:
        sentence_words = re.findall(r'\w+', sentence.lower())
        sentence_score = sum(word_freq[word] for word in sentence_words)
        sentence_scores.append((sentence, sentence_score))
    
    ranked_sentences = [sentence for sentence, score in sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]]
    
    summary = '. '.join(ranked_sentences)
    print(calculate_rouge_score(summary, text))
    return summary

def calculate_rouge_score(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(summary, reference)
    return scores