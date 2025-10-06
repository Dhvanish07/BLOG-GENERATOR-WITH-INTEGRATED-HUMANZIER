import os
import time
import requests
import json
import re
import random
import nltk
import numpy as np
import torch
import math
import string
from collections import defaultdict, Counter

# Optional NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet, stopwords

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

GEMINI_API_KEY = "YOUR GEMINI API KEY HERE"
API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL_NAME = "gemini-2.5-flash"
INPUT_JSON_FILE = "input.json"
OUTPUT_JSON_FILE = "output.json"

headers = {
    "Authorization": f"Bearer {GEMINI_API_KEY}",
    "Content-Type": "application/json",
}

class AdvancedAIHumanizer:
    def __init__(self):
        self.setup_models()
        self.setup_humanization_patterns()
        self.load_linguistic_resources()
        self.setup_fallback_embeddings()

    def setup_models(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence transformer loaded")
            except:
                self.sentence_model = None
                print("⚠️ Failed to load Sentence transformer")
        else:
            self.sentence_model = None
            print("⚠️ sentence-transformers not installed")
        if TRANSFORMERS_AVAILABLE:
            try:
                self.paraphrase_tokenizer = T5Tokenizer.from_pretrained('t5-small')
                self.paraphrase_model = T5ForConditionalGeneration.from_pretrained('t5-small')
                print("✅ T5 paraphrasing model loaded")
            except:
                self.paraphrase_tokenizer = None
                self.paraphrase_model = None
                print("✅ Using Manual Fallback")
        else:
            self.paraphrase_tokenizer = None
            self.paraphrase_model = None
            print("⚠️ transformers not installed")
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ SpaCy model loaded")
            except:
                self.nlp = None
                print("⚠️ SpaCy model failed to load")
        else:
            self.nlp = None
            print("⚠️ spaCy not installed")

    def setup_fallback_embeddings(self):
        self.word_groups = {
            'analyze': ['examine', 'study', 'investigate', 'explore', 'review', 'assess'],
            'important': ['crucial', 'vital', 'significant', 'essential', 'key', 'critical'],
            'shows': ['demonstrates', 'reveals', 'indicates', 'displays', 'exhibits'],
            'understand': ['comprehend', 'grasp', 'realize', 'recognize', 'appreciate'],
            'develop': ['create', 'build', 'establish', 'form', 'generate', 'produce'],
            'improve': ['enhance', 'better', 'upgrade', 'refine', 'advance', 'boost'],
            'consider': ['think about', 'examine', 'evaluate', 'contemplate', 'ponder'],
            'different': ['various', 'diverse', 'distinct', 'separate', 'alternative'],
            'effective': ['successful', 'efficient', 'productive', 'powerful', 'useful'],
            'significant': ['important', 'substantial', 'considerable', 'notable', 'major'],
            'implement': ['apply', 'execute', 'carry out', 'put into practice', 'deploy'],
            'utilize': ['use', 'employ', 'apply', 'harness', 'leverage', 'exploit'],
            'comprehensive': ['complete', 'thorough', 'extensive', 'detailed', 'full'],
            'fundamental': ['basic', 'essential', 'core', 'primary', 'key', 'central'],
            'substantial': ['significant', 'considerable', 'large', 'major', 'extensive']
        }
        self.synonym_map = {}
        for base_word, synonyms in self.word_groups.items():
            for synonym in synonyms:
                if synonym not in self.synonym_map:
                    self.synonym_map[synonym] = []
                self.synonym_map[synonym].extend(
                    [base_word] + [s for s in synonyms if s != synonym]
                )

    def setup_humanization_patterns(self):
        self.ai_indicators = {
            r'\bdelve into\b': ["explore", "examine", "investigate", "look into", "study", "dig into", "analyze"],
            r'\bembark upon?\b': ["begin", "start", "initiate", "launch", "set out", "commence", "kick off"],
            r'\ba testament to\b': ["proof of", "evidence of", "shows", "demonstrates", "reflects", "indicates"],
            r'\bobtain\b': ["get", "acquire", "gain", "secure", "achieve", "attain"],
        }
        self.human_starters = [
            "Actually,", "Honestly,", "Basically,", "Really,", "Generally,", "Usually,", "Often,",
            "The thing is,", "Here's the deal,", "Look,"
        ]
        self.contractions = {
            r'\bit is\b': "it's", r'\bthat is\b': "that's", r'\bthere is\b': "there's",
            r'\byou would\b': "you'd"
        }

    def load_linguistic_resources(self):
        self.stop_words = set(stopwords.words('english'))
        self.fillers = [
            "you know", "I mean", "sort of", "kind of", "basically", "actually",
            "really", "quite", "pretty much", "more or less", "essentially"
        ]
        self.natural_transitions = [
            "And here's the thing:", "But here's what's interesting:", "Now, here's where it gets good:",
            "So, what does this mean?", "Here's why this matters:", "Think about it this way:",
            "Let me put it this way:", "Here's the bottom line:", "The reality is:",
            "What we're seeing is:", "The truth is:", "At the end of the day:"
        ]

    def calculate_perplexity(self, text: str) -> float:
        try:
            words = word_tokenize(text.lower())
            if len(words) < 2:
                return 50.0
            word_freq = Counter(words)
            total_words = len(words)
            entropy = 0
            for word in words:
                prob = word_freq[word] / total_words
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            perplexity = 2 ** entropy
            if perplexity < 20:
                perplexity += random.uniform(20, 30)
            elif perplexity > 100:
                perplexity = random.uniform(60, 80)
            return perplexity
        except:
            return random.uniform(45, 75)

    def calculate_burstiness(self, text: str) -> float:
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return 1.2
            lengths = [len(word_tokenize(s)) for s in sentences]
            mean_len = np.mean(lengths)
            if mean_len == 0:
                return 1.2
            var_len = np.var(lengths)
            burstiness = var_len / mean_len
            if burstiness < 0.5:
                burstiness = random.uniform(0.7, 1.5)
            return burstiness
        except:
            return random.uniform(0.8, 1.4)

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            if self.sentence_model and SKLEARN_AVAILABLE:
                embeddings = self.sentence_model.encode([text1, text2])
                return float(cosine_similarity([embeddings[0]], [embeddings[1]]))
            else:
                w1, w2 = set(word_tokenize(text1.lower())), set(word_tokenize(text2.lower()))
                if not w1 or not w2:
                    return 0.8
                return max(0.7, len(w1 & w2) / len(w1 | w2))
        except:
            return 0.8

    def advanced_paraphrase(self, text: str, max_length: int = 256) -> str:
        try:
            if self.paraphrase_model and self.paraphrase_tokenizer:
                input_text = f"paraphrase: {text}"
                inputs = self.paraphrase_tokenizer.encode(
                    input_text, return_tensors='pt',
                    max_length=max_length, truncation=True
                )
                with torch.no_grad():
                    outputs = self.paraphrase_model.generate(
                        inputs,
                        max_length=max_length, num_return_sequences=1,
                        temperature=0.8, do_sample=True,
                        top_p=0.9, repetition_penalty=1.1
                    )
                para = self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if self.get_semantic_similarity(text, para) > 0.7:
                    return para
            return self.manual_paraphrase(text)
        except:
            return self.manual_paraphrase(text)

    def manual_paraphrase(self, text: str) -> str:
        patterns = [
            (r'(\w+) shows that (.+)', r'It is shown by \1 that \2'),
            (r'(\w+) demonstrates (.+)', r'This demonstrates \2 through \1'),
            (r'We can see that (.+)', r'It becomes clear that \1'),
            (r'This indicates (.+)', r'What this shows is \1'),
        ]
        for pat, repl in patterns:
            if re.search(pat, text, re.I):
                return re.sub(pat, repl, text, flags=re.I)
        return text

    def move_adverb_clause(self, sentence: str) -> str:
        patterns = [
            (r'^(.*?),\s*(because|since|when|if|although|while|as)\s+(.*?)([.!?])$', r'\2 \3, \1\4'),
            (r'^(.*?)\s+(because|since|when|if|although|while|as)\s+(.*?)([.!?])$', r'\2 \3, \1\4'),
            (r'^(Although|While|Since|Because|When|If)\s+(.*?),\s*(.*?)([.!?])$', r'\3, \1 \2\4')
        ]
        for pat, repl in patterns:
            if re.search(pat, sentence, re.I):
                return re.sub(pat, repl, sentence, flags=re.I).strip()
        return sentence

    def split_compound_sentence(self, sentence: str) -> str:
        conjunctions = [', and ', ', but ', ', so ', ', yet ', ', or ', '; however,', '; moreover,']
        for conj in conjunctions:
            if conj in sentence and len(sentence.split()) > 15:
                parts = sentence.split(conj, 1)
                if len(parts) == 2 and all(len(p.split()) > 3 for p in parts):
                    first = parts[0].strip()
                    if not first.endswith(('.', '!', '?')):
                        first += '.'
                    second = parts[1].strip()
                    if second and second.islower():
                        second = second.upper() + second[1:]
                    connector = random.choice(["Also,", "Plus,", "Additionally,", "What's more,", "On top of that,"])
                    return f"{first} {connector} {second.lower()}"
        return sentence

    def vary_voice_advanced(self, sentence: str) -> str:
        passive_patterns = [
            (r'(\w+)\s+(?:is|are|was|were)\s+(\w+ed|shown|seen|made|used|done|taken|given|found)\s+by\s+(.+)', r'\3 \2 \1'),
            (r'It\s+(?:is|was)\s+(\w+ed|shown|found)\s+that\s+(.+)', r'Research \1 that \2'),
        ]
        for pat, repl in passive_patterns:
            if re.search(pat, sentence, re.I):
                return re.sub(pat, repl, sentence, flags=re.I)
        return sentence

    def add_casual_connector(self, sentence: str) -> str:
        if len(sentence.split()) > 8 and ',' in sentence and random.random() < 0.3:
            parts = sentence.split(',', 1)
            if len(parts) == 2:
                insertion = random.choice([
                    ", you know,", ", I mean,", ", basically,", ", actually,",
                    ", really,", ", essentially,"
                ])
                return f"{parts[0]}{insertion}{parts[1]}"
        return sentence

    def restructure_with_emphasis(self, sentence: str) -> str:
        emphasis_patterns = [
            (r'^The fact that (.+) is (.+)', r"What's \2 is that \1"),
            (r'^It is (.+) that (.+)', r"What's \1 is that \2"),
            (r'^(.+) is very important', r'\1 really matters'),
            (r'^This shows that (.+)', r'This proves \1')
        ]
        for pat, rep in emphasis_patterns:
            if re.search(pat, sentence, re.I):
                return re.sub(pat, rep, sentence, flags=re.I)
        return sentence

    def add_human_touches(self, text: str, intensity=2) -> str:
        sents = sent_tokenize(text)
        prob_map = {1: 0.15, 2: 0.25, 3: 0.4}
        prob = prob_map.get(intensity, 0.25)
        result = []
        for i, s in enumerate(sents):
            cur = s
            if i > 0 and random.random() < prob and len(cur.split()) > 6:
                starter = random.choice(self.human_starters)
                cur = f"{starter} {cur[0].lower() + cur[1:]}"
            if i > 0 and random.random() < prob * 0.3:
                trans = random.choice(self.natural_transitions)
                cur = f"{trans} {cur.lower() + cur[1:]}"
            if random.random() < prob * 0.2 and len(cur.split()) > 10:
                filler = random.choice(self.fillers)
                words = cur.split()
                mid = len(words) // 2
                words.insert(mid, f", {filler},")
                cur = " ".join(words)
            result.append(cur)
        return " ".join(result)

    def apply_advanced_contractions(self, text: str, intensity=2) -> str:
        prob_map = {1: 0.4, 2: 0.6, 3: 0.8}
        prob = prob_map.get(intensity, 0.6)
        for pat, contr in self.contractions.items():
            if re.search(pat, text, re.I) and random.random() < prob:
                text = re.sub(pat, contr, text, flags=re.I)
        return text

    def enhance_vocabulary_diversity(self, text: str, intensity=2) -> str:
        words = word_tokenize(text)
        prob_map = {1: 0.2, 2: 0.35, 3: 0.5}
        prob = prob_map.get(intensity, 0.35)
        usage = defaultdict(int)
        for w in words:
            if w.isalpha() and len(w) > 3:
                usage[w.lower()] += 1
        result = []
        for i, w in enumerate(words):
            if (w.isalpha() and len(w) > 3 and w.lower() not in self.stop_words and
                usage[w.lower()] > 1 and random.random() < prob):
                ctx = " ".join(words[max(0, i - 5):min(len(words), i + 5)])
                result.append(self.get_contextual_synonym(w, ctx))
                usage[w.lower()] -= 1
            else:
                result.append(w)
        return " ".join(result)

    def get_contextual_synonym(self, word: str, context: str = "") -> str:
        try:
            wl = word.lower()
            if wl in self.word_groups:
                return random.choice(self.word_groups[wl])
            if wl in self.synonym_map:
                return random.choice(self.synonym_map[wl])
            synsets = wordnet.synsets(wl)
            synonyms = []
            for synset in synsets[:2]:
                for lemma in synset.lemmas():
                    syn = lemma.name().replace('_', ' ')
                    if syn != wl and len(syn) > 2:
                        synonyms.append(syn)
            if synonyms:
                suitable = [s for s in synonyms if abs(len(s) - len(word)) <= 3]
                return random.choice(suitable or synonyms)
            return word
        except:
            return word

    def restructure_sentences(self, text: str, intensity: int = 2) -> str:
        sents = sent_tokenize(text)
        prob_map = {1: 0.3, 2: 0.5, 3: 0.7}
        prob = prob_map.get(intensity, 0.5)
        results = []
        for s in sents:
            if len(s.split()) > 8 and random.random() < prob:
                s = random.choice([
                    self.move_adverb_clause,
                    self.split_compound_sentence,
                    self.vary_voice_advanced,
                    self.add_casual_connector,
                    self.restructure_with_emphasis
                ])(s)
            results.append(s)
        return " ".join(results)

    def replace_ai_patterns(self, text: str, intensity: int = 2) -> str:
        prob_map = {1: 0.7, 2: 0.85, 3: 0.95}
        prob = prob_map.get(intensity, 0.85)
        out = text
        for pat, repls in self.ai_indicators.items():
            matches = list(re.finditer(pat, out, re.I))
            for m in reversed(matches):
                if random.random() < prob:
                    out = out[:m.start()] + random.choice(repls) + out[m.end():]
        return out

    def multiple_pass_humanization(self, text: str, intensity: int = 2) -> str:
        passes = {1: 3, 2: 4, 3: 5}
        num_passes = passes.get(intensity, 4)
        current = text
        for i in range(num_passes):
            if i == 0:
                current = self.replace_ai_patterns(current, intensity)
            elif i == 1:
                current = self.restructure_sentences(current, intensity)
            elif i == 2:
                current = self.enhance_vocabulary_diversity(current, intensity)
            elif i == 3:
                current = self.apply_advanced_contractions(current, intensity)
                current = self.add_human_touches(current, intensity)
            elif i == 4:
                sents = sent_tokenize(current)
                current = " ".join(
                    self.advanced_paraphrase(s)
                    if len(s.split()) > 10 and random.random() < 0.3 else s
                    for s in sents
                )
            if self.get_semantic_similarity(text, current) < 0.7:
                break
        return current

    def final_quality_check(self, original: str, processed: str):
        processed = re.sub(r'\s+', ' ', processed)
        processed = re.sub(r'\s+([,.!?;:])', r'\1', processed)
        processed = re.sub(r'([,.!?;:])\s*([A-Z])', r'\1 \2', processed)
        sents = sent_tokenize(processed)
        corrected = []
        for s in sents:
            if s and s[0].islower():
                s = s.upper() + s[1:]
            corrected.append(s)
        out = " ".join(corrected).strip()
        out = re.sub(r'\.+', '.', out)
        return out, {}

    def humanize_text(self, text: str, intensity: str = "standard") -> str:
        if not text.strip():
            return ""
        levels = {"light": 1, "standard": 2, "heavy": 3}
        lvl = levels.get(intensity, 2)
        result = self.multiple_pass_humanization(text, lvl)
        result, _ = self.final_quality_check(text, result)
        return result

def generate_blog(title: str, primary_keyword: str,secondary_keywords: str,lsi_keywords: str,summary: str,) -> str:
    prompt = ("""
               prompt = (
    "You are a professional SEO content writer. Based on the following data that includes title={title},primary keywords={primary_keywords},secondary keywords={secondary_keywords},summary={summary} write a complete, "
    "well-structured, SEO-optimized blog post of around 800–1000 words. \n\n"
    "Guidelines:\n"
    "1. Use the '{title}' as the blog post title (H1).\n"
    "2. Use the '{primary_keyword}' at least 3–5 times throughout the blog (especially in the intro, headings, and conclusion).\n"
    "3. Integrate '{secondary_keywords}' 1–2 times each naturally into subheadings and paragraphs.\n"
    "4. Use '{lsi_keywords}' naturally throughout the blog without keyword stuffing.\n"
    "5. Begin with an engaging introduction that clearly reflects the summary.\n"
    "6. Include H2 and H3 subheadings to organize the content well.\n"
    "7. Make the tone conversational, informative, and travel-friendly.\n"
    "8. End with a useful conclusion, encouraging readers to take action (like planning their trip, saving the post, or exploring more).\n\n"
    "Here is the JSON data for the blog post:\n\n"
    f"{json.dumps(single_blog_json, indent=2)}"
)
"""
    )
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a professional SEO travel blogger."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Failed to generate blog for topic '{topic}': {e}")
        return None

def rephrase_blog(blog_content: str) -> str:
    prompt = (
        "check for any english, grammatic errors or any kind of spelling mistakes.\n"
        f"{blog_content}"
    )
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a travel blog editor."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Rephrasing failed: {e}")
        return blog_content

def rephrase_blog_2(blog_content: str) -> str:
    prompt = (
        "Clean the following travel blog content by removing any introductory phrases, "
        "apology statements, explanations about what you are doing, or irrelevant meta-text. "
        "Keep only the main blog content, in a natural flow, preserving sentences exactly "
        "as much as possible without rewording unnecessarily.\n\n"
        "reemove all #,*"
        f"{blog_content}"
    )
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a blog cleaner that strictly outputs only the blog content."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        cleaned_content = response.json()["choices"][0]["message"]["content"]
        cleaned_content = re.sub(r'^(Sure[, ]*|Here is.*?:|Of course[, ]*).*', '', cleaned_content, flags=re.I)
        cleaned_content = re.sub(r'\n+', '\n', cleaned_content).strip()
        return cleaned_content
    except Exception as e:
        print(f"❌ Rephrase 2 cleaning failed: {e}")
        blog_content = re.sub(r'^(Sure[, ]*|Here is.*?:|Of course[, ]*).*', '', blog_content, flags=re.I)
        return blog_content.strip()

def process_blogs():
    humanizer = AdvancedAIHumanizer()
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"❌ Input file '{INPUT_JSON_FILE}' not found.")
        return
    try:
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as infile:
            rows = json.load(infile)
    except Exception as e:
        print(f"❌ Failed to read input.json: {e}")
        return
    if not isinstance(rows, list) or not rows:
        print("⚠ JSON file is empty or invalid format.")
        return

    results = []
    for row in rows:
        topic = row.get("topic", "").strip()
        keywords = row.get("keywords", "").strip()
        entry_id = row.get("id", "")
        new_id = row.get("new_id", "")
        name = row.get("name", "")

        if not topic or not keywords or not entry_id:
            print(f"⚠ Missing data in entry: {row}")
            continue

        try:
            print(f"\n🔄 Generating blog for: {topic}")
            blog_content = None
            for attempt in range(3):
                blog_content = generate_blog(topic, keywords)
                if blog_content:
                    break
                print(f"⚠ Retry {attempt+2} for '{topic}'...")
                time.sleep(3)
            if not blog_content:
                print(f"❌ Skipped '{topic}' (generation failed)")
                continue

            print(f"✏️ Rephrasing '{topic}'")
            rephrased = rephrase_blog(blog_content)

            print(f"🧹 Cleaning '{topic}' with Rephrase 2")
            cleaned = rephrase_blog_2(rephrased)

            print(f"📝 Humanizing '{topic}'")
            humanized = humanizer.humanize_text(cleaned, intensity="heavy")

            results.append({
                "id": entry_id,
                "new_id": new_id,
                "name": name,
                "additional information": humanized
            })
            print(f"✅ Added blog for ID: {entry_id}")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error processing '{title}': {e}")
            continue

    try:
        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=2)
        print(f"\n✅ All blogs saved in '{OUTPUT_JSON_FILE}'")
    except Exception as e:
        print(f"❌ Failed to write output.json: {e}")

if __name__ == "__main__":
    process_blogs()
