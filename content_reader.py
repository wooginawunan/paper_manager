# %%
from pathlib import Path
import os
import fitz

from utils import load_all_papers, get_first_n_tokens_tiktoken
import ollama
import concurrent.futures
import pandas as pd
import argparse
from tqdm.rich import tqdm
from datetime import datetime
import pymupdf

DEFAULT_DOC_PATH = Path('/Users/nanwu/Desktop/readings')
ROOT_PATH = Path('/Users/nanwu/datasets/paper_manager/')

def doc_type_classifier(aspect_ratios, words_per_page):

    avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    avg_words_per_page = sum(words_per_page) / len(words_per_page)

    # Heuristic to differentiate
    if avg_aspect_ratio > 1.3 and avg_words_per_page < 100:
        return "presentation"
    elif avg_aspect_ratio < 1.3 and avg_words_per_page > 300:
        return "article"
    else:
        return "undetermined"

def analyze_pdf(file_path, n_pages_to_analyze=10):
    try:
        doc = fitz.open(file_path)
    except pymupdf.FileDataError:
        return None, None
    
    words_per_page = []
    aspect_ratios = []

    # only analyze the first 5 pages
    for page_num in range(min(n_pages_to_analyze, doc.page_count)):
        page = doc[page_num]
        # Get page dimensions (width and height)
        width, height = page.rect.width, page.rect.height
        aspect_ratio = width / height
        aspect_ratios.append(aspect_ratio)

        # Get text content
        text = page.get_text()
        if page_num == 0:
            font_page_text = text
        words = len(text.split())
        words_per_page.append(words)

    doc_type = doc_type_classifier(aspect_ratios, words_per_page)
    return doc_type, font_page_text


def get_file_info(file_idx, context):
    model = 'llama3.2:1b'
    prompt = """
    Extract the title, authors and year published from the following text: {font_page_text}

    Start your response with: title. authors (if available). year_published (if available).
    """
    info = ollama.generate(model=model, prompt=prompt.format(font_page_text=context))['response']
    return file_idx, info

    
def prepare_file_metadata(doc_path: Path, save_path: Path, n_pages_to_analyze: int = 5, n_tokens_to_context: int = 100):

    if not (save_path / 'paper_info.csv').exists():
        files = load_all_papers(doc_path)
        file_names = [file.name.strip('.pdf') for file in files]
        file_dates = [datetime.fromtimestamp(os.path.getmtime(file)).date() for file in tqdm(files, desc="Getting file dates")]
        file_types, font_page_texts = zip(*[analyze_pdf(file, n_pages_to_analyze=n_pages_to_analyze) \
            for file in tqdm(files, desc="Analyzing files")])
        file_contexts = [get_first_n_tokens_tiktoken(text, 'llama3.2:1b', n_tokens_to_context) if text is not None else None\
            for text in tqdm(font_page_texts, desc="Getting file contexts")]
        print(f"Analyzing {len(files)} files")

        file_metadata = pd.DataFrame({
            'file_idx': range(len(files)),
            'file_path': files, 
            'file_name': file_names, 
            'file_date': file_dates, 
            'doc_type': file_types, 
            'file_context': file_contexts,
            'file_info': [pd.NA] * len(files)
        })
        file_metadata.to_csv(save_path / 'paper_info.csv', index=True, escapechar='\\')
        
    else:
        file_metadata = pd.read_csv(save_path / 'paper_info.csv')
        print(f"Loaded {len(file_metadata)} files from {save_path / 'paper_info.csv'}")
        print(f"Remaining {file_metadata['file_info'].isna().sum()} files to process")

    return file_metadata

def main(args):
    doc_path = args.doc_path
    save_window = args.save_window
    n_pages_to_analyze = args.n_pages_to_analyze
    n_tokens_to_context = args.n_tokens_to_context

    save_path = ROOT_PATH / doc_path.name 
    save_path.mkdir(parents=True, exist_ok=True)

    file_metadata = prepare_file_metadata(
        doc_path, 
        save_path, 
        n_pages_to_analyze=n_pages_to_analyze, 
        n_tokens_to_context=n_tokens_to_context
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for _, row in tqdm(file_metadata.iterrows(), total=len(file_metadata), desc="Processing files"):
            file_idx, file_context = row['file_idx'], row['file_context']
            if pd.isna(row['file_info']) and file_context is not None:
                futures.append(executor.submit(get_file_info, file_idx, file_context))
            
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc="Processing results"):
            file_idx, info = future.result()
            file_metadata.at[file_idx, 'file_info'] = info
            if i % save_window == 0:
                file_metadata.to_csv(save_path / 'paper_info.csv', index=True)
        
    file_metadata.to_csv(save_path / 'paper_info.csv', index=True, escapechar='\\')
    print(f"Saved {len(file_metadata)} files to {save_path / 'paper_info.csv'}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--doc_path', type=Path, default=DEFAULT_DOC_PATH)
    argparser.add_argument('--save_window', type=int, default=10)
    argparser.add_argument('--n_pages_to_analyze', type=int, default=5)
    argparser.add_argument('--n_tokens_to_context', type=int, default=100)
    args = argparser.parse_args()
    main(args)
