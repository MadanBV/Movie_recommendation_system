import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import Scrollbar, Text, END

# Preload Dataset
CSV_PATH = "TMDB Movies DataSet.csv"
df = pd.read_csv(CSV_PATH)[['title', 'overview', 'release_date', 'vote_average', 'genre']].dropna()

# Compute TF-IDF and similarity
def get_recommendations(user_input, df, top_n=5):
    #Building Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['overview'])
    
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    recommendations = df.iloc[top_indices][['title', 'release_date', 'vote_average', 'genre', 'overview']]
    
    return recommendations

# GUI Functionality
def send_message():
    user_input = input_text.get("1.0", END).strip()
    if not user_input:
        result_text.insert(END, "Please enter a query.\n")   
        return
    recommendations = get_recommendations(user_input, df)
    result_text.delete('1.0', END)
    for _, row in recommendations.iterrows():
        result_text.insert(END, f"{row['title']} ({row['release_date']}), Rating: {row['vote_average']}\nGenre: {row['genre']}\n{row['overview']}\n\n")

# GUI Setup
root = tk.Tk()
root.title("Movie Recommendation System")

chatbox = Text(root, wrap=tk.WORD, state=tk.DISABLED, bg="#f5f5f5", font=("Arial", 12))
chatbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
scrollbar = Scrollbar(chatbox)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar.config(command=chatbox.yview)
chatbox.config(yscrollcommand=scrollbar.set)

result_text = Text(root, height=10, font=("Arial", 12), wrap=tk.WORD)
result_text.pack(padx=10, pady=(0, 10), fill=tk.BOTH)

input_text = Text(root, height=3, font=("Arial", 12), wrap=tk.WORD)
input_text.pack(padx=10, pady=(0, 10), fill=tk.BOTH)

send_button = tk.Button(root, text="Send", font=("Arial", 12), bg="#4caf50", fg="white", command=send_message)
send_button.pack(pady=(0, 10))

root.mainloop()
