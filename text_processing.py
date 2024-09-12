import whisper
import re
from textstat import flesch_reading_ease

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

def readability_score(text):
    return flesch_reading_ease(text)

def content_overlap(original_text, summary_text):
    original_words = set(re.findall(r'\w+', original_text.lower()))
    summary_words = set(re.findall(r'\w+', summary_text.lower()))
    overlap = len(original_words & summary_words) / len(original_words) * 100
    return overlap
