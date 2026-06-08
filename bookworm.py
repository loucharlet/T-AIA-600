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

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

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

    nlp = spacy.load("en_core_web_sm")

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

def summarize(text):

    text = strip_gutenberg(text)

    # découpagee en chapitres
    chapter_pattern = re.compile(
        r'\n\s*CHAPTER\s+([IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN'
        r'|EIGHT|NINE|TEN|ELEVEN|TWELVE)\b',
        re.IGNORECASE
    )

    splits = list(chapter_pattern.finditer(text))

    selected = []

    if splits:
        for i, match in enumerate(splits):
            start = match.end()
            end   = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            chapter_text = text[start:end].strip()

            # Prend les 3 premières phrases du chapitre (début = action principale)
            parser = PlaintextParser.from_string(chapter_text, Tokenizer("english"))
            sentences = list(parser.document.sentences)
            for s in sentences[:3]:
                line = str(s).strip()
                if 60 <= len(line) <= 200:
                    selected.append(line)
                    break  # 1 bonne phrase par chapitre suffit
    else:
        # Fallback : LSA sur tout le texte
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        selected = [str(s) for s in LsaSummarizer()(parser.document, 20)
                    if 60 <= len(str(s)) <= 200]

    # LSA sur les phrases sélectionnées pour garder les 5 meilleures
    if len(selected) > 5:
        combined = " ".join(selected)
        parser2  = PlaintextParser.from_string(combined, Tokenizer("english"))
        final    = LsaSummarizer()(parser2.document, 5)
        return " ".join(str(s) for s in final)

    return " ".join(selected[:5])


def main():
    parser = argparse.ArgumentParser(description="bookworm — analyse NLP de livres Gutenberg")
    parser.add_argument("--lexdiv",    metavar="book_id")
    parser.add_argument("--entities",  metavar="book_id")
    parser.add_argument("--summarize", metavar="book_id")

    args = parser.parse_args()

    if   args.lexdiv:    option, book_id = "lexdiv",    args.lexdiv
    elif args.entities:  option, book_id = "entities",  args.entities
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

    if   option == "lexdiv":    result = lexdiv(text)
    elif option == "entities":  result = entities(text)
    elif option == "summarize": result = summarize(text)

    cache_set(f"{option}_{book_id}", result)  # sauvegarde dans le cache
    print(result)


if __name__ == "__main__":
    main()