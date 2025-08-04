from typing import List
import re

class Chunker:
    def __init__(self, window_size: int = 3, stride: int = 1):
        self.window_size = window_size
        self.stride = stride

    def split_into_sentences(self, text: str) -> List[str]:
        # Basic sentence splitting (can be replaced with spaCy or nltk)
        sentence_endings = re.compile(r'(?<=[.!?\n]) +')
        return [s.strip() for s in sentence_endings.split(text) if s.strip()]

    def create_chunks(self, text: str) -> List[str]:
        sentences = self.split_into_sentences(text)
        chunks = []
        for i in range(0, len(sentences), self.stride):
            chunk = sentences[i:i + self.window_size]
            if chunk:
                chunks.append(" ".join(chunk))
            if i + self.window_size >= len(sentences):
                break
        return chunks
