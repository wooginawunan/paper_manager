# %%
from pathlib import Path
import pandas as pd
import shutil
from tqdm.rich import tqdm
ROOT_PATH = Path('/Users/nanwu/datasets/paper_manager/readings')

file_metadata = pd.read_csv(ROOT_PATH / 'paper_info.csv')
file_metadata = file_metadata.dropna(subset=['doc_type'], inplace=False)

# %%

def create_folder_structure(file_metadata: pd.DataFrame):
    file_types = file_metadata['doc_type'].unique()

    for file_type in file_types:
        save_path = ROOT_PATH / file_type
        save_path.mkdir(parents=True, exist_ok=True)

    print(f"Created folder structure for {len(file_types)} file types")

def move_files_to_local(file_metadata: pd.DataFrame):
    for _, row in tqdm(file_metadata.iterrows(), total=len(file_metadata), desc="Moving files to local"):
        file_path, file_name, file_type = row['file_path'], row['file_name'], row['doc_type']
        file_name = file_name + '.pdf'
        save_path = ROOT_PATH / file_type / file_name
        try:
            shutil.move(file_path, save_path)
        except FileNotFoundError:
            if not save_path.exists():
                print(f"File {file_path} not yet downloaded by icloud")

def update_file_metadata(file_metadata: pd.DataFrame):
    file_metadata['local_file_path'] = file_metadata[['file_name', 'doc_type']].apply(
        lambda x: ROOT_PATH / x['doc_type'] / (x['file_name']+'.pdf'), axis=1)
    file_metadata.to_csv(ROOT_PATH / 'paper_info.csv', index=False)
# %%
create_folder_structure(file_metadata)
move_files_to_local(file_metadata)
update_file_metadata(file_metadata)
# %%
