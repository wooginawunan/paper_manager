"""
Main script to process and organize academic papers.
"""
import os
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import concurrent.futures
import pandas as pd
from tqdm.rich import tqdm
import shutil
import fitz

from utils import (
    load_all_papers, 
    get_first_n_tokens_tiktoken, 
    read_pdf,
    find_duplicate_files,
    delete_duplicates,
    info_extraction_with_ollama
)


ROOT_PATH = Path('/Users/nanwu/datasets/paper_manager/')
DEFAULT_DOC_PATH = ROOT_PATH / 'recent'

class DocType(Enum):
    """Enum for the type of document."""
    PRESENTATION = "presentation"
    ARTICLE = "article"
    BOOK = "book"
    OTHER = "other"

@dataclass
class FileFeatures:
    """Dataclass for the features of a file."""
    avg_aspect_ratio: float
    avg_words_per_page: float
    num_pages: int

def doc_type_classifier(features: FileFeatures) -> DocType:
    """Classify the type of document based on the features."""
    avg_aspect_ratio = features.avg_aspect_ratio
    avg_words_per_page = features.avg_words_per_page
    num_pages = features.num_pages

    # Heuristic to differentiate
    if avg_aspect_ratio > 1.3 and avg_words_per_page < 100:
        # Presentation: 16:9 aspect ratio and less than 100 words per page
        return DocType.PRESENTATION
    elif avg_aspect_ratio < 1.3 and avg_words_per_page > 300:
        # 4:3 aspect ratio and more than 300 words per page
        if num_pages > 50:
            # Book: more than 50 pages
            return DocType.BOOK
        else:
            # Article: 4:3 aspect ratio and less than 50 pages
            return DocType.ARTICLE
    elif num_pages > 50:
        # Book: more than 50 pages
        return DocType.BOOK
    else:
        # Other: other types of documents
        return DocType.OTHER

def get_file_features(doc: fitz.Document, n_pages_to_analyze=10):
    """Get the features of a file."""
    num_pages = doc.page_count
    
    words_per_page = []
    aspect_ratios = []

    # only analyze the first 5 pages
    for page_num in range(min(n_pages_to_analyze, num_pages)):
        page = doc[page_num]
        # Get page dimensions (width and height)
        width, height = page.rect.width, page.rect.height
        aspect_ratio = width / height
        aspect_ratios.append(aspect_ratio)

        # Get text content
        text = page.get_text()
        words = len(text.split())
        words_per_page.append(words)

    return FileFeatures(
        avg_aspect_ratio=sum(aspect_ratios) / len(aspect_ratios),
        avg_words_per_page=sum(words_per_page) / len(words_per_page),
        num_pages=num_pages
    )


def get_file_types(files: list[Path], n_pages_to_analyze: int = 5):
    """Refine the file types of a list of files."""
    docs = [read_pdf(file) for file in tqdm(files, desc="Opening files")]
    file_features = [get_file_features(doc, n_pages_to_analyze) \
        if doc is not None else None for doc in tqdm(docs, desc="Getting file features")]
    file_types = [doc_type_classifier(features).value \
        if features is not None else None for features in tqdm(file_features, desc="Classifying file types")]
    return file_types


def load_file_metadata_from_files(
        doc_path: Path, 
        n_pages_to_analyze: int = 5,    
        n_tokens_to_context: int = 100, 
        model: str = 'llama3.2:1b'
    ):
    """Load the metadata of a list of files."""
    files = load_all_papers(doc_path)
    file_names = [file.name.strip('.pdf') for file in files]
    file_dates = [datetime.fromtimestamp(os.path.getmtime(file)).date() for file in tqdm(files, desc="Getting file dates")]

    docs = [read_pdf(file) for file in tqdm(files, desc="Opening files")]
    file_features = [get_file_features(doc, n_pages_to_analyze) \
        if doc is not None else None for doc in tqdm(docs, desc="Getting file features")]
    file_types = [doc_type_classifier(features).value \
        if features is not None else None for features in tqdm(file_features, desc="Classifying file types")]

    front_page_texts = [doc[0].get_text() if doc is not None else None for doc in docs]
    file_contexts = [get_first_n_tokens_tiktoken(text, model, n_tokens_to_context) if text is not None else None
        for text in tqdm(front_page_texts, desc="Getting file contexts")]

    print(f"Analyzing {len(files)} files")

    file_metadata = pd.DataFrame({
        'file_idx': range(len(files)),
        'file_path': files, 
        'file_name': file_names, 
        'file_date': file_dates, 
        'file_type': file_types, 
        'file_context': file_contexts,
        'file_info': [pd.NA] * len(files),
        'local_file_path': [pd.NA] * len(files)
    })
    return file_metadata


def create_folder_structure(save_path: Path, file_types: list[str]):
    """Create the folder structure for a list of file types."""
    for file_type in file_types:
        if file_type is not None and pd.notna(file_type):
            (save_path / file_type).mkdir(parents=True, exist_ok=True)

    print(f"Created folder structure for {len(file_types)} file types")


def move_files_to_local(file_metadata: pd.DataFrame):
    """Move the files to the local folder."""
    for i, row in tqdm(file_metadata.iterrows(), total=len(file_metadata), desc="Moving files to local"):
        file_path, file_type, local_file_path = row['file_path'], row['file_type'], row['local_file_path']
        if not Path(file_path).exists():
            print(f"File {file_path} not found")
            continue

        if pd.isna(local_file_path):
            file_metadata.at[i, 'local_file_path'] =  ROOT_PATH / file_type  / file_path.name
            shutil.move(file_path, ROOT_PATH / file_type / file_path.name)
            print(f"Moved {file_path} to {ROOT_PATH / file_type / file_path.name}")
        elif Path(local_file_path).exists():
            print(f"File {file_path} already moved to {local_file_path}")
        else:
            print(f"File {file_path} is missing at {local_file_path}")
            print(f"Removing local file path")
            file_metadata.at[i, 'local_file_path'] = pd.NA

    return file_metadata


def update_file_locations(file_metadata: pd.DataFrame, current_file_types: list[str]):
    """Update the file locations if the file type changed."""
    # update only if the file type changed and not None
    file_to_update = file_metadata[file_metadata['file_type'] != current_file_types]
    file_to_update = file_to_update[file_to_update['file_type'].notna()]
    for i, row in tqdm(file_to_update.iterrows(), total=len(file_to_update), desc="Updating file locations"):
        file_type, local_file_path = row['file_type'], row['local_file_path']
        
        print(f"Updating {local_file_path} to {file_type}")
        new_file_path = local_file_path.replace(current_file_types[i], file_type)
        row['local_file_path'] = new_file_path
        if Path(new_file_path).exists():
            print(f"File {new_file_path} already exists")
        elif Path(local_file_path).exists():
            shutil.move(local_file_path, new_file_path)
            print(f"Moved {local_file_path} to {new_file_path}")
        else:
            print(f"File {local_file_path} not found")

    return file_metadata


def main(args):
    """Main function to process and organize academic papers."""
    save_path = ROOT_PATH / 'readings'
    save_path.mkdir(parents=True, exist_ok=True)

    doc_path = args.doc_path
    n_pages_to_analyze = args.n_pages_to_analyze
    n_tokens_to_context = args.n_tokens_to_context
    model = args.model
    save_window = args.save_window

    # Load or create initial file metadata
    file_metadata = _load_or_create_metadata(
        save_path, doc_path, n_pages_to_analyze, n_tokens_to_context, model)

    # Refresh file list if requested
    if args.refresh_file_list:
        file_metadata = _refresh_file_list(
            file_metadata, save_path, doc_path, n_pages_to_analyze, n_tokens_to_context, model)

    # Clean duplicates if requested 
    if args.clean_duplicates:
        file_metadata = _clean_duplicate_files(file_metadata, save_path)

    # Extract titles if requested
    if args.title_extraction:
        file_metadata = _extract_titles(file_metadata, save_path, model, save_window)

    # Refine file types if requested
    if args.file_type_refinement:
        file_metadata = _refine_file_types(file_metadata, save_path, n_pages_to_analyze)


def _load_or_create_metadata(
    save_path: Path, 
    doc_path: Path, 
    n_pages_to_analyze: int, 
    n_tokens_to_context: int, 
    model: str
) -> pd.DataFrame:
    """Load existing metadata or create new from files."""
    metadata_path = save_path / 'paper_info.csv'
    
    if metadata_path.exists():
        file_metadata = pd.read_csv(metadata_path, index_col=0)
        print(f"Loaded {len(file_metadata)} files from {metadata_path}")
        print(f"Remaining {file_metadata['file_info'].isna().sum()} files to process")
        
        # Fill in missing file types with 'article'
        file_metadata['file_type'] = file_metadata['file_type'].fillna('article')

        # Convert file dates to datetime
        file_metadata['file_date'] = file_metadata['file_date'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d').date()
        )

        # Reset file indices
        file_metadata['file_idx'] = range(len(file_metadata))
    else:
        # Create new metadata from files
        file_metadata = load_file_metadata_from_files(
            doc_path, n_pages_to_analyze, 
            n_tokens_to_context, model
        )
        file_metadata.to_csv(metadata_path, index=True, escapechar='\\')
        print(f"Saved {len(file_metadata)} files to {metadata_path}")
    
    return file_metadata


def _refresh_file_list(
    file_metadata: pd.DataFrame, 
    save_path: Path, 
    doc_path: Path, 
    n_pages_to_analyze: int, 
    n_tokens_to_context: int, 
    model: str
) -> pd.DataFrame:
    """Refresh the file list with new files."""
    new_file_metadata = load_file_metadata_from_files(
        doc_path, n_pages_to_analyze,
        n_tokens_to_context, model
    )
    print(f"Loaded {len(new_file_metadata)} new files from {doc_path}")
    
    # Concatenate new files to existing metadata
    file_metadata = pd.concat([file_metadata, new_file_metadata], ignore_index=True)

    # Create folder structure
    create_folder_structure(save_path, file_metadata.file_type.unique())

    # Move files to local
    file_metadata = move_files_to_local(file_metadata)

    # Reset file indices
    file_metadata['file_idx'] = range(len(file_metadata))
    file_metadata.to_csv(save_path / 'paper_info.csv', index=True, escapechar='\\')
    
    print(f"Saved {len(file_metadata)} files to {save_path / 'paper_info.csv'}")
    return file_metadata


def _clean_duplicate_files(file_metadata: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """Remove duplicate files from the dataset."""
    files = file_metadata.local_file_path.to_list()
    file_dates = file_metadata.file_date.to_list()
    duplicate_pairs = find_duplicate_files(file_metadata.file_name.to_list())
    
    print("Number of duplicate pairs: ", len(duplicate_pairs))
    print("Duplicate pairs: ", duplicate_pairs)
    
    file_idx_to_delete = delete_duplicates(files, file_dates, duplicate_pairs)
    file_metadata = file_metadata.drop(index=file_idx_to_delete)
    file_metadata.to_csv(save_path / 'paper_info.csv', index=True, escapechar='\\')
    return file_metadata


def _extract_titles(
    file_metadata: pd.DataFrame, 
    save_path: Path, 
    model: str = 'llama3.2:1b',
    save_window: int = 10
) -> pd.DataFrame:
    """Extract titles from papers using Ollama."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for _, row in tqdm(file_metadata.iterrows(), total=len(file_metadata), desc="Processing files"):
            file_idx, file_context = row['file_idx'], row['file_context']
            if pd.isna(row['file_info']) and file_context is not None:
                futures.append(executor.submit(info_extraction_with_ollama, file_idx, file_context, model))
            
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc="Processing results"):
            file_idx, info = future.result()
            file_metadata.at[file_idx, 'file_info'] = info
            if i % save_window == 0:
                file_metadata.to_csv(save_path / 'paper_info.csv', index=True, escapechar='\\')

    file_metadata.to_csv(save_path / 'paper_info.csv', index=True, escapechar='\\')
    print(f"Saved {len(file_metadata)} files to {save_path / 'paper_info.csv'}")
    return file_metadata

def _refine_file_types(
    file_metadata: pd.DataFrame, 
    save_path: Path, 
    n_pages_to_analyze: int
) -> pd.DataFrame:
    """Refine file types and update folder structure."""
    if 'local_file_path' not in file_metadata.columns:
        print("No local file paths found, skipping file type refinement")
        return file_metadata
        
    current_file_types = file_metadata['file_type'].to_list()
    file_metadata['file_type'] = get_file_types(
        file_metadata.local_file_path, n_pages_to_analyze
    )

    create_folder_structure(save_path, file_metadata.file_type.unique())
    file_metadata = update_file_locations(file_metadata, current_file_types)
    file_metadata.to_csv(save_path / 'paper_info.csv', index=True, escapechar='\\')
    return file_metadata

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--doc_path', type=Path, default=DEFAULT_DOC_PATH)
    argparser.add_argument('--save_window', type=int, default=10)
    argparser.add_argument('--n_pages_to_analyze', type=int, default=5)
    argparser.add_argument('--n_tokens_to_context', type=int, default=100)
    argparser.add_argument('--clean_duplicates', action='store_true')
    argparser.add_argument('--title_extraction', action='store_true')
    argparser.add_argument('--file_type_refinement', action='store_true')
    argparser.add_argument('--refresh_file_list', action='store_true')
    argparser.add_argument('--model', type=str, default='llama3.2:1b')
    args = argparser.parse_args()

    main(args)
