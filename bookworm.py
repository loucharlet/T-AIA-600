"""
Lexical diversity metrics
Project Gutenberg book #11
"""

from collections import Counter
import urllib.request
import re


def get_book(book_id):

    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"

    response = urllib.request.urlopen(url)

    return response.read().decode("utf-8")


def lexdiv(text):

    text = text.lower()

    #si pas lettre, remplacé par espace
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    unique_words = set(words)  #supprime doublons

    word_counts = Counter(words)

    tok = len(words)  # nombre total de mots
    typ = len(unique_words)  # nb de mots différents

    hap = 0
    for word, count in word_counts.items():
        if count == 1:
            hap += 1  # nb de mots apparaissant 1 seule fois

    ttr = typ / tok  # nombre de mots différents / nb total de mots
    mwl = sum(len(word) for word in words) / tok  #longueur moyenne des mots
    mwf = tok / typ  # nombre total de mots /nombre de mots différents

    result = {
        "tok": tok,
        "typ": typ,
        "hap": hap,
        "ttr": round(ttr, 4),
        "mwl": round(mwl, 4),
        "mwf": round(mwf, 4)
    }

    return result


text = get_book(11)

result = lexdiv(text)

print(result)