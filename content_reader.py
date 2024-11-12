# %%
from pathlib import Path
import os
import fitz

from utils import load_all_papers
import ollama
import concurrent.futures
import pandas as pd
import argparse
from tqdm.rich import tqdm

DEFAULT_DOC_PATH = Path('/Users/nanwu/Desktop/readings')

ROOT_PATH = Path('./data')
# %%

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
    doc = fitz.open(file_path)
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


def get_file_info(file_path, prompt, model='llama3.2:1b'):
    doc_type, font_page_text = analyze_pdf(file_path, n_pages_to_analyze=5)
    info = ollama.generate(model=model, prompt=prompt.format(font_page_text=font_page_text))['response']
    return file_path, doc_type, info

def main(doc_path: Path):
    model = 'llama3.2:1b'
    prompt = """
    Extract the title, authors and year published from the following text: {font_page_text}

    You return text: [title. authors (if available). year_published (if available).]
    """

    files = load_all_papers(doc_path)[:3]
    file_names = [file.name.strip('.pdf') for file in files]
    file_dates = [os.path.getmtime(file) for file in files]

    print(f"Analyzing {len(files)} files")

    file_metadata = pd.DataFrame({'file_path': files, 'file_name': file_names, 'file_date': file_dates})
    print(file_metadata)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_file_info, file_path, prompt, model) for file_path in tqdm(files, desc="Submitting tasks")]
        results = list(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing results"))

    parsed_results = [result.result() for result in tqdm(results, desc="Parsing results")]
    file_contents = pd.DataFrame(parsed_results, columns=['file_path', 'doc_type', 'info'])
    file_info = pd.merge(file_metadata, file_contents, on='file_path')

    save_path = ROOT_PATH / doc_path.name
    save_path.mkdir(exist_ok=True)
    file_info.to_csv(save_path / 'paper_info.csv', index=True)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--doc_path', type=Path, default=DEFAULT_DOC_PATH)
    args = argparser.parse_args()
    main(args.doc_path)