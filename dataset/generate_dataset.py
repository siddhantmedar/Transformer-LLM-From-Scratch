from datasets import load_dataset
import re

def clean_pg19(text):
    """Clean PG-19 text by removing Gutenberg boilerplate and extra whitespace."""
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(start_marker):end_idx]
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load PG-19 (first 20 books for ~10–20 MB)
pg19_dataset = load_dataset("pg19", split="train[:20]")
pg19_texts = [clean_pg19(entry['text']) for entry in pg19_dataset if entry['text']]
with open('pg19.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(pg19_texts))

def clean_wikitext(text):
    """Clean WikiText by removing markup and extra whitespace."""
    text = re.sub(r'=\s*[^=]+\s*=', ' ', text)  # Remove headings
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else ''

wikitext_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
wikitext_texts = [clean_wikitext(entry['text']) for entry in wikitext_dataset if entry['text']]
with open('wikitext.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(wikitext_texts))

def clean_arxiv(text):
    """Clean Arxiv text by removing LaTeX, special characters, and extra whitespace."""
    text = re.sub(r'\\[a-zA-Z]+{[^}]*}', '', text)  # Remove LaTeX
    text = re.sub(r'[\n\r]+', ' ', text)  # Remove newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else ''

arxiv_dataset = load_dataset("arxiv_dataset", split="train[:10000]")  # ~10 MB
arxiv_texts = [clean_arxiv(entry['abstract']) for entry in arxiv_dataset if entry['abstract']]
with open('arxiv.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(arxiv_texts))

# Combine datasets into input.txt
def combine_datasets(shakespeare_file, pg19_file, wikitext_file, arxiv_file, output_file):
    with open(shakespeare_file, 'r', encoding='utf-8') as f:
        shakespeare_text = f.read()
    with open(pg19_file, 'r', encoding='utf-8') as f:
        pg19_text = f.read()
    with open(wikitext_file, 'r', encoding='utf-8') as f:
        wikitext_text = f.read()
    with open(arxiv_file, 'r', encoding='utf-8') as f:
        arxiv_text = f.read()
    
    # Combine with newlines for separation
    combined_text = f"{shakespeare_text}\n{pg19_text}\n{wikitext_text}\n{arxiv_text}"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)

# Run combination
combine_datasets(
    shakespeare_file='dataset/tiny_shakespeare.txt',  # Current Tiny Shakespeare
    pg19_file='pg19.txt',
    wikitext_file='wikitext.txt',
    arxiv_file='arxiv.txt',
    output_file='dataset/input.txt'  # Overwrite with combined dataset
)