import os
from dotenv import load_dotenv

load_dotenv()
print("Token bulundu mu:", os.getenv("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") is not None)
print("Token:", os.getenv("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"))
