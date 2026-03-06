import os
from dotenv import load_dotenv
import random

class KeyRotator:
    def __init__(self):
        load_dotenv()
        self.gemini_keys = [
            os.getenv('GEMINI_KEY_1'),
            os.getenv('GEMINI_KEY_2'),
            os.getenv('GEMINI_KEY_3'),
            os.getenv('GEMINI_KEY_4'),
            os.getenv('GEMINI_API_KEY')
        ]
        self.gemini_keys = [k for k in self.gemini_keys if k]
        
        self.mistral_keys = [
            os.getenv('MISTRAL_API_KEY'),
            os.getenv('CODESTRAL_API_KEY')
        ]
        self.mistral_keys = [k for k in self.mistral_keys if k]
        
        self.current_gemini_idx = 0
        self.current_mistral_idx = 0

    def get_gemini_key(self, rotate=True):
        if not self.gemini_keys:
            return None
        key = self.gemini_keys[self.current_gemini_idx]
        if rotate:
            self.current_gemini_idx = (self.current_gemini_idx + 1) % len(self.gemini_keys)
        return key

    def get_mistral_key(self, rotate=True):
        if not self.mistral_keys:
            return None
        key = self.mistral_keys[self.current_mistral_idx]
        if rotate:
            self.current_mistral_idx = (self.current_mistral_idx + 1) % len(self.mistral_keys)
        return key

    def get_random_gemini(self):
        return random.choice(self.gemini_keys) if self.gemini_keys else None
