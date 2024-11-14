# %%
from pathlib import Path
from transformers import AutoTokenizer
import tiktoken

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