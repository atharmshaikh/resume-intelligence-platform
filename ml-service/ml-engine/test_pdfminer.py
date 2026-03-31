from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pathlib import Path
import time

path = "sample_resumes/MBIT_IT_RanaAyanMaiyuddin - AYAN RANA.pdf"
print(f"Testing {path}...")
laparams = LAParams(line_margin=0.4, word_margin=0.1, char_margin=2.0)

start = time.time()
try:
    text = extract_text(path, laparams=laparams)
    print(f"Success! Time: {time.time() - start:.2f}s")
    print(f"Text length: {len(text)}")
except Exception as e:
    print(f"Error: {e}")
