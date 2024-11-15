# %%
from pathlib import Path
import os
from transformers import AutoTokenizer
import tiktoken
import fitz
import ollama

ollama_model_mapper_huggingface = {
    'llama3.2:1b': 'meta-llama/Llama-3.2-1B-Instruct'
}

ollama_model_mapper_tiktoken = {
    'llama3.2:1b': 'cl100k_base'
}


def get_first_n_tokens_tiktoken(text: str, model_name: str, n: int) -> str:
    """Returns the first n tokens of a text string."""
    encoding_name = ollama_model_mapper_tiktoken[model_name]
    encoding = tiktoken.get_encoding(encoding_name)
    token_integers = encoding.encode(text)
    token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers[:n]]
    text = b''.join(token_bytes).decode('utf-8', errors='ignore')
    return text


def get_num_tokens_tiktoken(text: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = ollama_model_mapper_tiktoken[model_name]
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)


def get_num_tokens_huggingface(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(ollama_model_mapper_huggingface[model_name])
    tokens = tokenizer.apply_chat_template([text])
    return len(tokens)


def load_all_papers(doc_path: Path):
    files = list(doc_path.glob('*.pdf'))
    files.extend(doc_path.glob('*/*.pdf'))
    files.extend(doc_path.glob('*/*/*.pdf'))
    return files

def read_pdf(file_path: Path):
    try:
        import pymupdf
        try:
            return fitz.open(file_path)
        except (pymupdf.FileDataError, pymupdf.FileNotFoundError):
            return None

    except ImportError:
        try:
            return fitz.open(file_path)
        except fitz.fitz.FileNotFoundError:
            return None

def diff_count(name1: str, name2: str) -> int:
    return sum(1 for a, b in zip(name1, name2) if a != b)

def find_duplicate_files(file_names: list[str]) -> list[tuple[int, int]]:
    # Remove file extensions for comparison
    duplicates = []
    # Remove file extensions for comparison
    for i, name1 in enumerate(file_names):
        for j, name2 in enumerate(file_names[i + 1:], start=i + 1):
            # Check if one name is contained within another
            if name1.startswith(name2) \
                or name2.startswith(name1):
                if diff_count(name1, name2) <= 10:
                    duplicates.append((i, j))
    
    return duplicates

def delete_duplicates(
        files: list[Path], 
        file_dates: list[float], 
        duplicate_pairs: list[tuple[int, int]]
    ) -> list[int]:
    # Delete the duplicate with the smaller date
    file_idx_to_delete = []
    for original, duplicate in duplicate_pairs:
        file_to_delete = files[original] if file_dates[original] < file_dates[duplicate] else files[duplicate]
        print(f"Deleting {file_to_delete}")
        file_idx_to_delete.append(original if file_dates[original] < file_dates[duplicate] else duplicate)
        try:
            os.remove(file_to_delete)
        except FileNotFoundError:
            print(f"File {file_to_delete} not found")
    return file_idx_to_delete

def info_extraction_with_ollama(file_idx, context, model: str = 'llama3.2:1b'):
    """Get the info of a file."""
    prompt = """
    Extract the title, authors and year published from the following text: {font_page_text}

    Start your response with: title. authors (if available). year_published (if available).
    """
    info = ollama.generate(model=model, prompt=prompt.format(font_page_text=context))['response']
    return file_idx, info