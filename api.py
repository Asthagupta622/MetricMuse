from flask import Flask, request, jsonify, make_response
import nltk
from textblob import TextBlob
import textstat
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

@app.route('/')
def home():
    return " MetricMuse API is running"

@app.route('/analyze-text', methods=['POST', 'OPTIONS'])
def analyze_text():
    if request.method == 'OPTIONS':
        return make_response('', 204)

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # NLP functions
    def grammatical_error_percentage(text):
        blob = TextBlob(text)
        original_words = text.split()
        corrected_words = str(blob.correct()).split()
        errors = sum(1 for o, c in zip(original_words, corrected_words) if o != c)
        return round((errors / len(original_words) * 100), 2) if original_words else 0

    def calculate_readability(text):
        return textstat.flesch_reading_ease(text)

    def average_sentence_length(text):
        sentences = textstat.sentence_count(text)
        return textstat.lexicon_count(text) / sentences if sentences else 0

    def find_repetitive_words(text):
        stop_words = set(stopwords.words('english'))
        words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words and len(w) > 3]
        freq_dist = nltk.FreqDist(words)
        repetitive = {w: c for w, c in freq_dist.items() if c > 1}
        return len(repetitive), ', '.join([f"{w}: {c}" for w, c in repetitive.items()])

    def assess_generic_content(text):
        filler_words = ['basically', 'various', 'very', 'things', 'stuff', 'it is important', 'clearly', 'obviously', 'however', 'therefore', 'furthermore', 'in conclusion']
        words = word_tokenize(text)
        total = len(words)
        filler_count = sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text.lower())) for w in filler_words)
        return round((filler_count / total * 100), 2) if total > 0 else 0

    def detect_ai_content(text):
        generic_score = 0 if assess_generic_content(text) else 1
        total_words = len(word_tokenize(text))
        repetitive_count, _ = find_repetitive_words(text)
        repetitive_density = repetitive_count / total_words if total_words > 0 else 0
        sentence_len = average_sentence_length(text)
        sentence_score = 1 if sentence_len < 15 else 0
        return round(((generic_score + repetitive_density + sentence_score) / 3) * 100, 2)

    # Build metrics response
    metrics = {
        "Grammatical Error Percentage": grammatical_error_percentage(text),
        "Readability Score": calculate_readability(text),
        "Average Sentence Length": average_sentence_length(text),
        "Repetitive Words Count": find_repetitive_words(text)[0],
        "Repetitive Words List": find_repetitive_words(text)[1],
        "AI Content Percentage": detect_ai_content(text),
        "Generic Content Percentage": assess_generic_content(text)
    }

    return jsonify(metrics)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render’s assigned port
    app.run(host='0.0.0.0', port=port)
