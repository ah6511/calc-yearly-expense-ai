---
name: extract-expense-from-chase-bank-stmt
description: Use this skill to analyze Chase bank statements. It should analyze Chase bank statement pdf file under 'pdf' folder, extract withdrawals and then save the withdrawals to csv to 'csv' folder using same name as pdf file.
---

# Chase Bank Statement Analysis Guide

## Overview

Read PDF files from the `pdf/` folder, extract withdrawals, and save them as CSV files in the `csv/` folder using the same base filename.

## Step 1 — Run the extraction script directly

**IMPORTANT: Do NOT run any shell commands before the script — no `ls pdf/`, no `pip install`, no `pip show`, no file checks.** The script already handles PDF discovery (`glob.glob("pdf/*.pdf")`) and installs `pypdf` inline. Your very first action must be to execute the complete script from Step 3 in a single shell command. Skip Steps 1 and 2 entirely and go straight to Step 3.

## Step 2 — Understand the Chase PDF text layout (CRITICAL)

Chase bank statement PDFs have a **non-intuitive text layout** when extracted with pypdf:

- **In the visual PDF**, all section headers (`DEPOSITS AND ADDITIONS`, `ELECTRONIC WITHDRAWALS`, etc.) appear before their transactions — as expected. **In the pypdf-extracted text stream**, however, withdrawal section headers (e.g., `ELECTRONIC WITHDRAWALS`) appear **after** the transactions that belong to that section. `DEPOSITS AND ADDITIONS` keeps its normal order in the extracted text (header first, then transactions with `*start*`/`*end*` markers).
- The text also contains internal boundary markers in lowercase: `*start*deposits and additions` and `*end*deposits and additions`.
- Transaction amounts may or may not have a leading `$` (e.g., `$162.00` or `155.28`).
- Multi-line transaction descriptions are joined onto the same logical line in the extracted text.

**Reliable section boundaries in the raw extracted text (verified order of appearance):**

```
DEPOSITS AND ADDITIONS          ← summary table header (ignore)
CHECKING SUMMARY                ← appears early in document (ignore as end boundary)
*start*deposits and additions   ← start of deposit transactions
...deposit transactions...
*end*deposits and additions     ← end of deposit transactions  ← USE AS START BOUNDARY
...ATM & DEBIT CARD transactions...
ATM & DEBIT CARD WITHDRAWALS    ← section label appears AFTER its own transactions
...ELECTRONIC transactions...   ← (may be absent in some statements)
ELECTRONIC WITHDRAWALS          ← section label appears AFTER its own transactions
...OTHER transactions...        ← (may be absent in some statements)
OTHER WITHDRAWALS               ← section label appears AFTER its own transactions ← USE FURTHEST AS END BOUNDARY
```

**WARNING:** `CHECKING SUMMARY` appears *before* `*end*deposits and additions` in the raw text — do NOT use it as the end boundary or the extraction will fall back to the entire document and include deposits.

**Use `*end*deposits and additions` as the start boundary. For the end boundary, scan for all three known section headers (`ATM & DEBIT CARD WITHDRAWALS`, `ELECTRONIC WITHDRAWALS`, `OTHER WITHDRAWALS`) after the start, and use the furthest one found.** This handles statements that have only ATM withdrawals, only electronic withdrawals, or a combination.

## Step 3 — Complete extraction script

Use this script as-is. It handles all known Chase PDF quirks:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf"], check=True)

import re
import csv
import os
import glob
from pypdf import PdfReader

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_withdrawals(text):
    # Isolate the withdrawal block using reliable boundary markers.
    # Start: *end*deposits and additions
    # End: the FURTHEST occurrence of any known withdrawal section header.
    # Section headers appear AFTER their transactions in Chase PDFs, so the
    # furthest header marks the natural end of the withdrawal block.
    # This handles statements with only ATM, only Electronic, or both.
    start_marker = "*end*deposits and additions"
    start_idx = text.find(start_marker)
    if start_idx == -1:
        raise ValueError(
            "Could not find start marker '*end*deposits and additions'. "
            "Check that the PDF is a Chase checking statement."
        )

    end_candidates = [
        "ATM & DEBIT CARD WITHDRAWALS",
        "ELECTRONIC WITHDRAWALS",
        "OTHER WITHDRAWALS",
    ]
    end_idx = -1
    for marker in end_candidates:
        idx = text.find(marker, start_idx)
        if idx != -1 and idx > end_idx:
            end_idx = idx

    if end_idx == -1:
        raise ValueError(
            "Could not find any withdrawal section header after the start marker. "
            "Check that the PDF is a Chase checking statement."
        )
    withdrawal_block = text[start_idx + len(start_marker):end_idx]

    # Match lines: MM/DD <description> <amount>
    # Amount may have optional leading $ and optional thousands commas
    pattern = re.compile(
        r'^(\d{2}/\d{2})\s+(.+?)\s+\$?([\d,]+\.\d{2})\s*$',
        re.MULTILINE
    )

    withdrawals = []
    for match in pattern.finditer(withdrawal_block):
        date, description, amount_str = match.group(1), match.group(2).strip(), match.group(3)
        amount = float(amount_str.replace(",", ""))
        withdrawals.append({"date": date, "description": description, "amount": amount})

    return withdrawals

def save_to_csv(withdrawals, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["date", "description", "amount"])
        writer.writeheader()
        writer.writerows(withdrawals)

# Process all PDFs
for pdf_path in glob.glob("pdf/*.pdf"):
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    csv_path = f"csv/{basename}.csv"

    text = extract_text(pdf_path)
    withdrawals = extract_withdrawals(text)
    save_to_csv(withdrawals, csv_path)
    print(f"Saved {len(withdrawals)} withdrawals → {csv_path}")
```

## Withdrawal inclusion and exclusion rules (APPLY AFTER SCRIPT)

After the script saves the CSV, **re-read the CSV and remove any rows** whose `description` (case-insensitive) contains any of the exclude keywords below, then overwrite the CSV with the filtered rows.

| Rule | Action |
|------|--------|
| Category is `ATM & DEBIT CARD WITHDRAWALS`, `ELECTRONIC WITHDRAWALS`, or `OTHER WITHDRAWALS` | **Include** |
| Category is `DEPOSITS AND ADDITIONS` | **Exclude** |
| Description contains `Payment To`, `Purchase`, `Bill`, `City of Naperville`, `Autopay`, `Pymt`, or `Loan` | **Always include** (do not deduplicate) |
| Description contains any of: `Payment From`, `Payroll`, `Deposit`, `College Savings`, `Fid Bkg`, `Vanguard`, `Tuition`, `Rent`, `Pl*Americancampu`, `Pl*Paylease`, `Simplebills`, `Irs Treas`, `State of Ill`, `AT&T Services`, `529 Dir ACH` | **Always exclude** (takes priority over include rule) |

## Output CSV format

| Column | Type | Notes |
|--------|------|-------|
| `date` | string | `MM/DD` format |
| `description` | string | Full description, no truncation |
| `amount` | float | No `$`, no commas (e.g., `162.0`) |