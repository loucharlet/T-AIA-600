# Bookworm

Bookworm is a lightweight NLP engine designed to analyze books from Project Gutenberg and generate structured book cards.

The project provides several NLP features through a command-line interface:

* Lexical diversity analysis
* Topic extraction
* Named Entity Recognition (NER)
* Book summarization
* Book similarity recommendation
* Complete book card generation

The goal is to transform raw literary text into structured metadata that can help publishers, editors, and readers quickly understand a book.

---

# Setup

## Requirements

* Python 3.10+
* spaCy
* NLTK
* NumPy
* scikit-learn

## Installation

Clone the repository:

```bash
git clone git@github.com:loucharlet/T-AIA-600.git
cd T-AIA-600
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install nltk spacy numpy scikit-learn
```

Download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Run the program:

```bash
python bookworm.py --help
```

---

# Usage

## Lexical Diversity

Computes vocabulary richness metrics.

```bash
python bookworm.py --lexdiv 11
```

Example output:

```python
{
    "tok": 27439,
    "typ": 2579,
    "hap": 1107,
    "ttr": 0.094,
    "mwl": 3.9412,
    "mwf": 10.6394
}
```

---

## Named Entity Recognition

Extracts characters and locations from a book.

```bash
python bookworm.py --entities 11
```

Example output:

```python
{
    "characters": ["Alice", "Queen", "King"],
    "locations": ["Wonderland"]
}
```

---

## Topic Modeling

Extracts the 10 most representative words from each quarter of the book.

```bash
python bookworm.py --topics 11
```

Example output:

```python
{
    1: ["alice", "rabbit", "..."],
    2: ["queen", "cat", "..."],
    3: [...],
    4: [...]
}
```

---

## Summarization

Generates a lightweight summary based on extracted topics and book category.

```bash
python bookworm.py --summarize 11
```

Example output:

```text
"Alice's Adventures in Wonderland" by Lewis Carroll is a children and young adult novel...
```

---

## Similar Books

Returns the 5 most similar books from the dataset.

```bash
python bookworm.py --similar 11
```

Example output:

```python
[
    "Through the Looking-Glass",
    "Peter Pan",
    "The Wonderful Wizard of Oz",
    "The Secret Garden",
    "The Jungle Book"
]
```

---

## Book Card

Aggregates every NLP feature into a single structured dictionary.

```bash
python bookworm.py --card 11
```

Example output:

```python
{
    "info": {...},
    "lexdiv": {...},
    "topics": {...},
    "entities": {...},
    "summary": "...",
    "similar": [...]
}
```

---

# NLP Pipeline

The project follows a classical NLP pipeline.

```text
Raw Gutenberg Book
        │
        ▼
Remove Gutenberg Header/Footer
        │
        ▼
Tokenization
        │
        ▼
Stopword Removal
        │
        ▼
Lemmatization
        │
        ▼
Feature Extraction
        │
        ├── Lexical Diversity
        ├── Topic Extraction
        ├── Named Entities
        ├── Summarization
        └── Similarity Analysis
```

## Cleaning

Project Gutenberg headers and footers are removed before processing.

## Tokenization

The text is converted to lowercase, punctuation is removed, and words are split into tokens.

## Stopword Removal

Common English words such as:

```text
the, and, is, of, ...
```

are removed using NLTK stopwords.

## Normalization

Words are normalized using WordNet lemmatization.

Example:

```text
running → run
cars → car
```

## Vectorization

For similarity analysis, books are converted into TF-IDF vectors and compared using cosine similarity.

---

# Methodology

## Lexical Diversity

The lexical diversity module computes:

* Total tokens
* Unique tokens
* Hapax legomena
* Type-Token Ratio
* Mean Word Length
* Mean Word Frequency

These metrics provide a quick estimation of vocabulary richness.

---

## Topic Extraction

The book is divided into four sections.

For each section:

1. Text cleaning
2. Stopword removal
3. Frequency analysis
4. Top-10 keywords extraction

This lightweight approach provides interpretable themes while remaining computationally inexpensive.

---

## Named Entity Recognition

Named entities are extracted using spaCy's `en_core_web_sm` model.

The system keeps:

* PERSON → characters
* GPE / LOC → locations

Additional filtering removes noisy entities.

---

## Summarization

The summarization module uses a lightweight keyword-based approach.

Process:

1. Extract topics
2. Remove generic words
3. Detect book category
4. Generate a summary template

Advantages:

* Fast
* No heavy models
* Easy to explain

Limitations:

* Less precise than transformer-based summarization
* Relies on extracted keywords

Alternative approaches considered:

* TextRank
* LSA summarization
* Transformer-based summarization (rejected because heavy models are not allowed)

---

## Similarity Analysis

Books are compared using:

```text
TF-IDF Vectorization
        +
Cosine Similarity
```

This method is lightweight, deterministic, and effective for literary recommendation tasks.

---

# Caching System

To avoid recomputing expensive operations, results are stored in:

```text
cache/
```

using pickle files.

Downloaded books are also stored locally in:

```text
BOOKS/
```

to avoid downloading them multiple times.

---

# Rubric Skills Coverage

## 1. Text Preprocessing

✔ Cleaning

✔ Tokenization

✔ Stopword Removal

✔ Lemmatization

---

## 2. NLP Feature Extraction

✔ Lexical Diversity

✔ Topic Extraction

✔ Named Entity Recognition

✔ Summarization

---

## 3. Vectorization & Similarity

✔ TF-IDF

✔ Cosine Similarity

✔ Recommendation System

---

## 4. Software Engineering

✔ Command Line Interface

✔ Error Handling

✔ Caching

✔ Modular Functions

---

## 5. Book Card Generation

✔ Metadata Extraction

✔ Aggregation of NLP Features

✔ Structured Dictionary Output

---

# Project Structure

```text
.
├── bookworm.py
├── README.md
├── BOOKS/
├── cache/
└── .venv/
```

---

# Authors

Lou Charlet

Epitech – T-AIA-600
