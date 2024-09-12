from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_word_cloud(text, file_name):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(file_name, format='png')
    plt.close()
