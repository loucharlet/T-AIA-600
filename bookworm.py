"""
Lexical diversity metrics
Project Gutenberg book #11
"""

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.corpus import stopwords
from collections import Counter
from pathlib import Path
import urllib.request
import argparse
import pickle
import re
import sys
import spacy
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

'''global model cache'''
_NLP_MODEL = None
_TFIDF_VECTORIZER = None

def get_nlp_model():
    global _NLP_MODEL
    if _NLP_MODEL is None:
        _NLP_MODEL = spacy.load("en_core_web_sm")
    return _NLP_MODEL

BOOKS = {
    "11":   {"title": "Alice's Adventures in Wonderland",     "author": "Lewis Carroll"},
    "12":   {"title": "Through the Looking-Glass",             "author": "Lewis Carroll"},
    "16":   {"title": "Peter Pan",                             "author": "James Matthew Barrie"},
    "55":   {"title": "The Wonderful Wizard of Oz",            "author": "Lyman Frank Baum"},
    "113":  {"title": "The Secret Garden",                     "author": "Frances Hodgson Burnett"},
    "120":  {"title": "Treasure Island",                       "author": "Robert Louis Stevenson"},
    "236":  {"title": "The Jungle Book",                       "author": "Rudyard Kipling"},
    "108":  {"title": "The Return of Sherlock Holmes",         "author": "Arthur Conan Doyle"},
    "1661": {"title": "The Adventures of Sherlock Holmes",     "author": "Arthur Conan Doyle"},
    "863":  {"title": "The Mysterious Affair at Styles",       "author": "Agatha Christie"},
    "35":   {"title": "The Time Machine",                      "author": "H.G. Wells"},
    "36":   {"title": "The War of the Worlds",                 "author": "H.G. Wells"},
    "84":   {"title": "Frankenstein",                          "author": "Mary Shelley"},
    "164":  {"title": "Twenty Thousand Leagues under the Sea", "author": "Jules Verne"},
    "345":  {"title": "Dracula",                               "author": "Bram Stoker"},
}

# dossiers pour le cache
BOOKS_DIR = Path("BOOKS")   # textes téléchargés
CACHE_DIR  = Path("cache")  # résultats calculés

BOOKS_DIR.mkdir(exist_ok=True)  # crée si n'existe pas
CACHE_DIR.mkdir(exist_ok=True)


def cache_get(key):
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    return None  # pas encore en cache


def cache_set(key, value):
    path = CACHE_DIR / f"{key}.pkl"
    pickle.dump(value, open(path, "wb"))



def get_book(book_id):
    book_file = BOOKS_DIR / f"{book_id}.txt"

    if book_file.exists():
        return book_file.read_text(encoding="utf-8")  # déjà téléchargé

    print(f"Téléchargement du livre {book_id}...")
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"

    # User-Agent obligatoire sinon Gutenberg renvoie 403
    req      = urllib.request.Request(url, headers={"User-Agent": "bookworm/1.0"})
    response = urllib.request.urlopen(req)
    text     = response.read().decode("utf-8")

    book_file.write_text(text, encoding="utf-8")  # sauvegarde pour la prochaine fois
    return text


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


def entities(text):

    nlp = get_nlp_model() 

    characters = []
    locations = []

    for i in range(0, len(text), 100000):
        doc = nlp(text[i:i + 100000])

        for ent in doc.ents:

            if ent.label_ == "PERSON":
                characters.append(ent.text.strip())

            if ent.label_ in ["GPE", "LOC"]:
                locations.append(ent.text.strip())

    result = {
        "characters": sorted(set(characters)),
        "locations": sorted(set(locations))
    }

    return result



def strip_gutenberg(text):
    start = re.search(r"\*\*\* ?START OF TH[EI]S? PROJECT GUTENBERG", text, re.IGNORECASE)
    end   = re.search(r"\*\*\* ?END OF TH[EI]S? PROJECT GUTENBERG",   text, re.IGNORECASE)
    if start:
        text = text[text.find("\n", start.end()) + 1:]
    if end:
        end2 = re.search(r"\*\*\* ?END OF TH[EI]S? PROJECT GUTENBERG", text, re.IGNORECASE)
        if end2:
            text = text[:end2.start()]
    return text.strip()

def topics(text):
    """Divise le texte en 4 sections et extrait les 10 mots-clés de chaque section
    via fréquence TF simple après suppression des stopwords."""
    clean = strip_gutenberg(text)
    stop_words = set(stopwords.words("english"))

    section_size = len(clean) // 4
    sections = {
        1: clean[0:section_size],
        2: clean[section_size:2*section_size],
        3: clean[2*section_size:3*section_size],
        4: clean[3*section_size:],
    }

    result = {}
    for section_id, section_text in sections.items():
        section_text = section_text.lower()
        section_text = re.sub(r"[^a-z\s]", " ", section_text)
        words = [w for w in section_text.split() if len(w) > 3 and w not in stop_words]
        top_10 = [word for word, _ in Counter(words).most_common(10)]
        result[section_id] = top_10

    return result

BAD_ENTITY_WORDS = {
    "chapter", "ebook", "project", "gutenberg", "illustration",
    "contents", "copyright", "release", "language", "title"
}

BAD_TOPIC_WORDS = {
    "said", "alice", "would", "could", "little", "like", "went",
    "came", "come", "looked", "thought", "know", "time", "one"
}

def clean_entity_name(name):
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[^A-Za-z\s'\-]", "", name).strip()
    return name

def ranked_entities(text):
    nlp = get_nlp_model()
    persons = Counter()
    places = Counter()

    for i in range(0, len(text), 100000):
        doc = nlp(text[i:i + 100000])

        for ent in doc.ents:
            name = clean_entity_name(ent.text)
            low = name.lower()

            if len(name) < 3:
                continue
            if any(bad in low for bad in BAD_ENTITY_WORDS):
                continue

            if ent.label_ == "PERSON":
                persons[name] += 1
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                places[name] += 1

    return {
        "characters": [name for name, _ in persons.most_common(10)],
        "locations": [name for name, _ in places.most_common(10)]
    }

def clean_topics(topic_data):
    result = []
    for words in topic_data.values():
        for word in words:
            word = word.lower()
            if word not in BAD_TOPIC_WORDS and len(word) > 3:
                result.append(word)
    return list(dict.fromkeys(result))

def get_book_header_info(text, book_id):
    header = text[:5000]

    title_match = re.search(r"^Title:\s*(.+)$", header, re.MULTILINE)
    author_match = re.search(r"^Author:\s*(.+)$", header, re.MULTILINE)

    title = (
        title_match.group(1).strip()
        if title_match
        else BOOKS.get(book_id, {}).get("title", "Unknown title")
    )

    author = (
        author_match.group(1).strip()
        if author_match
        else BOOKS.get(book_id, {}).get("author", "Unknown author")
    )

    return title, author, None

def pick_item(items, index, fallback):
    if items and len(items) > index:
        return items[index]
    return fallback

def book_genre(book_id):
    children = {"11", "12", "16", "55", "113", "120", "236"}
    mystery = {"108", "834", "863", "1661", "61262", "69087", "70114"}
    scifi = {"35", "36", "84", "159", "164", "345", "68283"}

    if book_id in children:
        return "children"
    if book_id in mystery:
        return "mystery"
    if book_id in scifi:
        return "scifi"
    return "general"

def summarize(text, book_id):
    clean = strip_gutenberg(text)
    title, author, release = get_book_header_info(text, book_id)

    entity_data = ranked_entities(clean)
    topic_data = topics(clean)

    characters = entity_data["characters"]
    locations = entity_data["locations"]
    topic_words = clean_topics(topic_data)

    char1 = pick_item(characters, 0, "the main character")
    char2 = pick_item(characters, 1, "another important character")
    place = pick_item(locations, 0, "an important setting")
    theme1 = pick_item(topic_words, 0, "adventure")
    theme2 = pick_item(topic_words, 1, "conflict")

    genre = book_genre(book_id)

    templates = {
        "children": (
            f'"{title}" is a children and young adult book written by {author}. '
            f'The story follows {char1}, who moves through {place} and meets {char2}. '
            f'The book is shaped by themes such as {theme1} and {theme2}, creating a playful and imaginative journey.'
        ),
        "mystery": (
            f'"{title}" is a crime and mystery book written by {author}. '
            f'The story follows {char1}, connected with {char2}, through events linked to {theme1} and {theme2}. '
            f'The setting of {place} helps create a tense atmosphere built around clues, conflict, and investigation.'
        ),
        "scifi": (
            f'"{title}" is a science-fiction or fantasy book written by {author}. '
            f'The story follows {char1} through {place}, where unusual events involve {theme1} and {theme2}. '
            f'Through these elements, the book builds a speculative world shaped by danger, discovery, and transformation.'
        ),
        "general": (
            f'"{title}" was written by {author}. '
            f'The story follows {char1}, with important figures such as {char2}, around {place}. '
            f'Its main themes include {theme1} and {theme2}, giving a compact overview of the book.'
        )
    }

    return templates.get(genre, templates["general"])

    
def main():
    parser = argparse.ArgumentParser(description="bookworm — analyse NLP de livres Gutenberg")
    parser.add_argument("--lexdiv",    metavar="book_id")
    parser.add_argument("--entities",  metavar="book_id")
    parser.add_argument("--topics",    metavar="book_id")
    parser.add_argument("--summarize", metavar="book_id")

    args = parser.parse_args()

    if   args.lexdiv:    option, book_id = "lexdiv",    args.lexdiv
    elif args.entities:  option, book_id = "entities",  args.entities
    elif args.topics:    option, book_id = "topics",    args.topics
    elif args.summarize: option, book_id = "summarize", args.summarize
    else:
        parser.print_help()
        sys.exit(1)

    # vérifie le cache d'abord
    cached = cache_get(f"{option}_{book_id}")
    if cached is not None:
        print(cached)
        return

    try:
        text = get_book(book_id)
    except Exception as e:
        print(f"Erreur : impossible de récupérer le livre {book_id}. {e}")
        sys.exit(1)

    try:
        if   option == "lexdiv":    result = lexdiv(text)
        elif option == "entities":  result = entities(text)
        elif option == "topics":    result = topics(text)
        elif option == "summarize": result = summarize(text, book_id)
    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    cache_set(f"{option}_{book_id}", result)
    print(result)

if __name__ == "__main__":
    main()