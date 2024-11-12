from pathlib import Path

def load_all_papers(doc_path: Path):
    files = list(doc_path.glob('*.pdf'))
    files.extend(doc_path.glob('*/*.pdf'))
    return files