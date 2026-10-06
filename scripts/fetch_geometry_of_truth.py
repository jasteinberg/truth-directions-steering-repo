"""Fetch Geometry-of-Truth datasets (Marks & Tegmark 2023) into the local data dir.
Source: https://github.com/saprmarks/geometry-of-truth  (datasets/, main branch).
A reproducible substitute for committing the CSVs. Run from any env (stdlib only).
Scaffolded with assistance from Claude (Anthropic)."""
import os
import sys
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "geometry_of_truth")
BASE = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets"
FILES = ["cities.csv", "neg_cities.csv", "larger_than.csv", "smaller_than.csv",
         "sp_en_trans.csv", "neg_sp_en_trans.csv", "common_claim_true_false.csv",
         "companies_true_false.csv", "counterfact_true_false.csv"]
os.makedirs(DATA, exist_ok=True)
failed = []
for f in FILES:
    try:
        urllib.request.urlretrieve(f"{BASE}/{f}", f"{DATA}/{f}"); print("ok", f)
    except OSError as e:                     # network or HTTP error: report every file, then fail
        print("FAIL", f, e)
        failed.append(f)
if failed:
    sys.exit(f"{len(failed)} of {len(FILES)} files failed: {', '.join(failed)}")
