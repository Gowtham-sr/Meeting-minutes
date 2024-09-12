import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from utils.audio_processing import convert_mp4_to_wav, split_wav_file
from utils.text_processing import transcribe_audio, readability_score, content_overlap
from utils.summarization import summarize_with_tfidf, summarize_with_lda, summarize_with_frequency, advanced_summarize_text
from utils.visualization import generate_word_cloud

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def summarize_mp4(mp4_file):
    wav_file = "output_audio.wav"
    convert_mp4_to_wav(mp4_file, wav_file)
    wav_chunks = split_wav_file(wav_file)
    
    full_transcription = ""
    for chunk in wav_chunks:
        transcribed_text = transcribe_audio(chunk)
        full_transcription += transcribed_text + " "
    
    tfidf_summary = summarize_with_tfidf(full_transcription)
    lda_summary = summarize_with_lda(full_transcription)
    freq_summary = summarize_with_frequency(full_transcription)
    advanced_summary = advanced_summarize_text(full_transcription)
    
    for chunk in wav_chunks:
        os.remove(chunk)
    
    tfidf_readability = readability_score(tfidf_summary)
    lda_readability = readability_score(lda_summary)
    freq_readability = readability_score(freq_summary)
    advanced_readability = readability_score(advanced_summary)
    
    tfidf_overlap = content_overlap(full_transcription, tfidf_summary)
    lda_overlap = content_overlap(full_transcription, lda_summary)
    freq_overlap = content_overlap(full_transcription, freq_summary)
    advanced_overlap = content_overlap(full_transcription, advanced_summary)

    generate_word_cloud(tfidf_summary, "tfidf_wordcloud.png")
    generate_word_cloud(lda_summary, "lda_wordcloud.png")
    generate_word_cloud(freq_summary, "freq_wordcloud.png")
    generate_word_cloud(advanced_summary, "advanced_wordcloud.png")
    
    print(f"TF-IDF Summary Readability Score: {tfidf_readability:.2f}")
    print(f"LDA Summary Readability Score: {lda_readability:.2f}")
    print(f"Frequency-Based Summary Readability Score: {freq_readability:.2f}")
    print(f"Advanced Summary Readability Score: {advanced_readability:.2f}")
    
    print(f"TF-IDF Summary Content Overlap: {tfidf_overlap:.2f}%")
    print(f"LDA Summary Content Overlap: {lda_overlap:.2f}%")
    print(f"Frequency-Based Summary Content Overlap: {freq_overlap:.2f}%")
    print(f"Advanced Summary Content Overlap: {advanced_overlap:.2f}%")
    
    with open("video_summary.txt", "w") as summary_file:
        summary_file.write(f"TF-IDF Summary:\n{tfidf_summary}\n\n")
        summary_file.write(f"LDA Summary:\n{lda_summary}\n\n")
        summary_file.write(f"Frequency-Based Summary:\n{freq_summary}\n\n")
        summary_file.write(f"Advanced Summary:\n{advanced_summary}\n")
    
    return tfidf_summary, lda_summary, freq_summary, advanced_summary

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            tfidf_summary, lda_summary, freq_summary, advanced_summary = summarize_mp4(file_path)
            return render_template('index.html', 
                                   tfidf_summary=tfidf_summary,
                                   lda_summary=lda_summary,
                                   freq_summary=freq_summary,
                                   advanced_summary=advanced_summary)
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
