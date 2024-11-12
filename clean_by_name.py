# %%
from pathlib import Path
import os
import argparse
from utils import load_all_papers

DEFAULT_DOC_PATH = Path('/Users/nanwu/Desktop/readings')

def remove_duplicates_by_name(files: list[Path]):
    return list(set(files))
# %%
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
    ):
    # Delete the duplicate with the smaller date
    for original, duplicate in duplicate_pairs:
        file_to_delete = files[original] if file_dates[original] < file_dates[duplicate] else files[duplicate]
        print(f"Deleting {file_to_delete}")
        try:
            os.remove(file_to_delete)
        except FileNotFoundError:
            print(f"File {file_to_delete} not found")

# Find and print duplicates

# %%
def main(doc_path: Path):
    files = load_all_papers(doc_path)
    file_names = [file.name.strip('.pdf') for file in files]
    file_dates = [os.path.getmtime(file) for file in files]

    duplicate_pairs = find_duplicate_files(file_names)
    print(f"Found {len(duplicate_pairs)} duplicates")
    delete_duplicates(files, file_dates, duplicate_pairs)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--doc_path', type=Path, default=DEFAULT_DOC_PATH)
    args = argparser.parse_args()
    main(args.doc_path)
# %%
