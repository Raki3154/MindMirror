from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import re
from datetime import datetime
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
CORS(app)

class TextAnalyzer:
    def __init__(self):
        self.filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'sort of', 'kind of']
        self.qualifier_words = ['maybe', 'perhaps', 'possibly', 'i think', 'i believe', 'probably', 'sort of', 'kind of']
        self.transition_words = ['however', 'therefore', 'furthermore', 'additionally', 'consequently', 'meanwhile', 'thus', 'hence']
        self.negative_words = ['unfortunately', 'unable', 'cannot', "won't", "can't", 'sorry', 'apologize', 'regret']
        self.extreme_words = ['always', 'never', 'completely', 'absolutely', 'perfectly', 'extremely', 'totally']

    def analyze_cognitive_load(self, text):
        """Analyze cognitive load based on speech patterns"""
        words = text.lower().split()
        sentences = nltk.sent_tokenize(text)
        word_count = len(words)
        sentence_count = len(sentences)
        
        score = 70  # Base score
        
        # Filler words analysis
        filler_count = sum(text.lower().count(filler) for filler in self.filler_words)
        if filler_count > 5:
            score -= 20
        elif filler_count > 2:
            score -= 10
        
        # Sentence complexity
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if avg_sentence_length > 25:
                score -= 15
            elif avg_sentence_length < 8:
                score -= 8
        
        # Repetition analysis
        unique_words = set(words)
        repetition_ratio = len(unique_words) / word_count if word_count > 0 else 1
        if repetition_ratio < 0.6:
            score -= 12
        
        # Hesitation patterns
        hesitation_patterns = len(re.findall(r'\.\.+|\-\-+', text))
        score -= hesitation_patterns * 3
        
        return max(30, min(95, score))

    def analyze_honesty(self, text):
        """Analyze honesty indicators in text"""
        text_lower = text.lower()
        words = text_lower.split()
        sentences = nltk.sent_tokenize(text)
        
        score = 65  # Base score
        
        # Qualifier analysis
        qualifier_count = sum(text_lower.count(qualifier) for qualifier in self.qualifier_words)
        if qualifier_count > 3:
            score -= 15
        elif qualifier_count > 1:
            score -= 8
        
        # Negative emotion words
        negative_count = sum(text_lower.count(negative) for negative in self.negative_words)
        if negative_count > 2:
            score -= 10
        
        # Extreme words (potential exaggeration)
        extreme_count = sum(text_lower.count(extreme) for extreme in self.extreme_words)
        if extreme_count > 2:
            score -= 8
        
        # Sentence complexity (overly complex might indicate deception)
        if sentences:
            complex_sentences = sum(1 for s in sentences if len(s.split()) > 20)
            complex_ratio = complex_sentences / len(sentences)
            if complex_ratio > 0.3:
                score -= 7
        
        # Consistency check (simple version)
        if len(sentences) > 2:
            first_half = ' '.join(sentences[:len(sentences)//2])
            second_half = ' '.join(sentences[len(sentences)//2:])
            if abs(len(first_half) - len(second_half)) > len(text) * 0.4:
                score -= 5
        
        return max(40, min(90, score))

    def analyze_idea_clarity(self, text):
        """Analyze clarity and organization of ideas"""
        sentences = nltk.sent_tokenize(text)
        words = text.split()
        word_count = len(words)
        sentence_count = len(sentences)
        
        score = 60  # Base score
        
        if sentence_count == 0:
            return 50
        
        # Transition words
        transition_count = sum(text.lower().count(transition) for transition in self.transition_words)
        score += min(20, transition_count * 4)
        
        # Sentence variety
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = max(sentence_lengths) - min(sentence_lengths) if sentence_lengths else 0
        if length_variance > 10:
            score += 8
        elif length_variance < 3:
            score -= 5
        
        # Paragraph structure simulation
        if sentence_count >= 3:
            score += 10
        
        # Readability (simple version)
        avg_sentence_length = word_count / sentence_count
        if 15 <= avg_sentence_length <= 25:
            score += 8
        elif avg_sentence_length > 30:
            score -= 7
        
        # Coherence indicators
        if text.lower().count('but') + text.lower().count('however') > 0:
            score += 5  # Shows contrast and complex thought
        
        return max(45, min(95, score))

    def get_reasons(self, category, score, text):
        """Generate detailed reasons for scores"""
        text_lower = text.lower()
        words = text_lower.split()
        sentences = nltk.sent_tokenize(text)
        
        if category == 'cognitiveLoad':
            filler_count = sum(text_lower.count(filler) for filler in self.filler_words)
            reasons = []
            
            if filler_count > 3:
                reasons.append(f"High frequency of filler words ({filler_count} instances)")
            elif filler_count > 0:
                reasons.append(f"Some filler words present ({filler_count} instances)")
            else:
                reasons.append("Minimal use of filler words")
            
            if sentences:
                avg_length = len(words) / len(sentences)
                if avg_length > 20:
                    reasons.append("Complex sentence structures requiring more processing")
                elif avg_length < 10:
                    reasons.append("Simple sentence structures")
                else:
                    reasons.append("Appropriate sentence complexity")
            
            hesitation = len(re.findall(r'\.\.+|\-\-+', text))
            if hesitation > 0:
                reasons.append(f"Hesitation patterns detected ({hesitation} instances)")
            
            return reasons[:3]
        
        elif category == 'honesty':
            qualifier_count = sum(text_lower.count(qualifier) for qualifier in self.qualifier_words)
            negative_count = sum(text_lower.count(negative) for negative in self.negative_words)
            reasons = []
            
            if qualifier_count > 2:
                reasons.append(f"Use of qualifying language ({qualifier_count} instances)")
            elif qualifier_count > 0:
                reasons.append("Some qualifying phrases present")
            else:
                reasons.append("Direct and confident language")
            
            if negative_count > 1:
                reasons.append(f"Negative emotion words used ({negative_count} instances)")
            
            extreme_count = sum(text_lower.count(extreme) for extreme in self.extreme_words)
            if extreme_count > 1:
                reasons.append("Use of absolute terms detected")
            
            return reasons[:3]
        
        elif category == 'ideaClarity':
            transition_count = sum(text_lower.count(transition) for transition in self.transition_words)
            reasons = []
            
            if transition_count > 0:
                reasons.append(f"Good use of transition words ({transition_count} instances)")
            else:
                reasons.append("Limited use of transition words")
            
            if len(sentences) > 2:
                reasons.append("Multiple sentences with developed ideas")
            elif len(sentences) > 0:
                reasons.append("Limited content for comprehensive analysis")
            
            if text_lower.count('but') + text_lower.count('however') > 0:
                reasons.append("Shows ability to present contrasting ideas")
            else:
                reasons.append("Consider adding more perspective variety")
            
            return reasons[:3]

analyzer = TextAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        username = data.get('username', 'Anonymous')
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided for analysis'}), 400
        
        # Perform analysis
        cognitive_score = analyzer.analyze_cognitive_load(text)
        honesty_score = analyzer.analyze_honesty(text)
        clarity_score = analyzer.analyze_idea_clarity(text)
        
        results = {
            'cognitiveLoad': {
                'score': round(cognitive_score),
                'reasons': analyzer.get_reasons('cognitiveLoad', cognitive_score, text)
            },
            'honesty': {
                'score': round(honesty_score),
                'reasons': analyzer.get_reasons('honesty', honesty_score, text)
            },
            'ideaClarity': {
                'score': round(clarity_score),
                'reasons': analyzer.get_reasons('ideaClarity', clarity_score, text)
            }
        }
        
        # Log analysis
        print(f"Analysis completed for {username}: "
              f"Cognitive={results['cognitiveLoad']['score']}%, "
              f"Honesty={results['honesty']['score']}%, "
              f"Clarity={results['ideaClarity']['score']}%")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'MindMirror Analysis API',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting MindMirror Analysis Server...")
    print("API available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
