import nltk
from transformers import PreTrainedTokenizer
from typing import Any, List, Dict, Optional
import warnings
import torch

from core import config
from core.data.tasks.mapping_task import MappingTask
from comet import download_model, load_from_checkpoint
# import sacrebleu

# Suppress warnings from COMET
warnings.filterwarnings("ignore", category=UserWarning)

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet as wn


# Global cache for COMET model (singleton pattern to avoid reloading)
_COMET_MODEL_CACHE = None


def get_comet_model():
    """Get or create a cached COMET model instance on GPU"""
    global _COMET_MODEL_CACHE
    if _COMET_MODEL_CACHE is None:
        model_path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL_CACHE = load_from_checkpoint(model_path)
        # Move to GPU if available
        if torch.cuda.is_available():
            _COMET_MODEL_CACHE = _COMET_MODEL_CACHE.cuda()
    return _COMET_MODEL_CACHE


class TranslationTask(MappingTask):

    def __init__(self, tokenizer: PreTrainedTokenizer, mapping_type: str, mapping_name: str, allow_prefix: bool = False):
        super().__init__(tokenizer, mapping_type, mapping_name, allow_prefix)

        # Use shared COMET model (lazy loading)
        self.comet_model = None

        # Language mapping for supported languages
        self.supported_languages = {
            "en": "eng",
            "fr": "fra",
            "it": "ita",
            "es": "spa",
            "ja": "jpn"
        }

    def _load_comet_model(self):
        """Load COMET model for evaluation (uses cached instance)"""
        if self.comet_model is None:
            self.comet_model = get_comet_model()
    
    @staticmethod
    def _get_synonyms(word: str, lang_to: str):
        lang = {
            "en": "eng",
            "fr": "fra",
            "it": "ita",
            "es": "spa",
        }[lang_to]
        synonyms = [word]
        for syn in wn.synsets(word, lang=lang):
            for lemma in syn.lemmas(lang=lang):
                synonyms.append(lemma.name())
        return synonyms

    def compare_outputs(self, output1: Any, output2: Any) -> bool:
        output1, output2 = output1.strip(), output2.strip()
        output_lang = self.mapping_name.split("_")[1]
        if output_lang == "ja":
           return output1 == output2
        synonyms1 = self._get_synonyms(output1, output_lang)
        synonyms2 = self._get_synonyms(output2, output_lang)
        return len(set(synonyms1) & set(synonyms2)) > 0
    
    
    def evaluate_with_comet(self, sources: List[str], predictions: List[str], references: List[str]) -> Dict[str, float]:
        # Lazy load COMET model if not already loaded
        if self.comet_model is None:
            self._load_comet_model()

        # Prepare data for COMET
        comet_input = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        # Calculate COMET score on GPU with larger batch size
        # gpus=1 means use 1 GPU, batch_size increased for H100
        comet_scores = self.comet_model.predict(comet_input, batch_size=64, gpus=1)
        print(len(comet_scores))
        print("翻訳前",sources)
        print("予測",predictions)
        print("正解",references)
        print("スコア",comet_scores)
        print("cometの平均スコア",sum(comet_scores.scores) / len(comet_scores.scores))
        print(max(comet_scores))
        print(min(comet_scores))
        return {
            "comet": sum(comet_scores.scores) / len(comet_scores.scores),
            "comet_scores": comet_scores.scores
        }
    