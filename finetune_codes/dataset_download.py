from datasets import load_dataset
import os

# Symlink uyarısını kapat
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Veri setini yükle (trust_remote_code=True şart)
dataset = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)
