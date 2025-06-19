# Blackcoffer Text Analysis Assignment

## Overview

This project extracts article text from URLs and performs comprehensive text analysis to compute 13 variables including sentiment scores, readability metrics, and word complexity measures.

## Dependencies Installation

### Step 1: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 2: Verify NLTK Data

The script will automatically download required NLTK data (punkt tokenizer) on first run.

## Project Structure

```
.
├── Input.xlsx                    # Input URLs and URL_IDs
├── index.py                      # Main analysis script
├── requirements.txt             # Python dependencies
├── MasterDictionary/           # Sentiment word lists
│   ├── negative-words.txt
│   └── positive-words.txt
├── StopWords/                  # Stop word collections
│   ├── StopWords_Auditor.txt
│   ├── StopWords_Currencies.txt
│   ├── StopWords_DatesandNumbers.txt
│   ├── StopWords_GenericLong.txt
│   ├── StopWords_Generic.txt
│   ├── StopWords_Geographic.txt
│   └── StopWords_Names.txt
├── Output Data Structure.xlsx   # Expected output format
└── Text Analysis.docx          # Variable definitions
```

## How to Run

### Step 1: Ensure Input File

Make sure `Input.xlsx` exists with columns:

- URL_ID
- URL

### Step 2: Run the Analysis

```bash
python index.py
```

### Step 3: Check Output

The script generates:

- `output_results.xlsx` - Main results file
- `output_results.csv` - CSV version for backup
- `extracted_articles/` - Directory with extracted text files (named by URL_ID)

## Solution Approach

### 1. Data Extraction

- Uses `requests` and `BeautifulSoup` for web scraping
- Implements multiple content selectors to find article text
- Removes headers, footers, navigation, and other non-article content
- Saves extracted text files with URL_ID as filename

### 2. Text Processing

- Loads positive/negative word dictionaries for sentiment analysis
- Combines all stop word files for comprehensive text cleaning
- Uses NLTK for sentence and word tokenization
- Implements custom syllable counting algorithm

### 3. Analysis Computation

**Sentiment Scores:**

- POSITIVE SCORE: Count of positive words after stop word removal
- NEGATIVE SCORE: Count of negative words after stop word removal
- POLARITY SCORE: (Positive - Negative) / (Positive + Negative + 0.000001)
- SUBJECTIVITY SCORE: (Positive + Negative) / (Total Words + 0.000001)

**Readability Metrics:**

- AVG SENTENCE LENGTH: Total words / Total sentences
- PERCENTAGE OF COMPLEX WORDS: (Complex words / Total words) × 100
- FOG INDEX: 0.4 × (Avg sentence length + Percentage complex words)
- COMPLEX WORD COUNT: Words with more than 2 syllables
- SYLLABLE PER WORD: Average syllables across all words

**Word Analysis:**

- WORD COUNT: Total cleaned words after stop word removal
- PERSONAL PRONOUNS: Count of I, we, my, ours, us
- AVG WORD LENGTH: Average character length of words

### 4. Error Handling

- Robust URL processing with timeout and retries
- Graceful handling of failed extractions
- Comprehensive logging for debugging
- Default scores for failed analyses

## Output Format

The output matches the structure in `Output Data Structure.xlsx`:

- URL_ID, URL (from input)
- All 13 computed text analysis variables

## Features

- **Multi-selector Content Extraction**: Uses various CSS selectors to find article content
- **Comprehensive Stop Word Removal**: Combines 7 different stop word lists
- **Custom Syllable Counter**: Handles English syllable patterns including silent 'e'
- **Robust Error Handling**: Continues processing even if individual URLs fail
- **Progress Logging**: Real-time feedback on processing status
- **Respectful Scraping**: 1-second delay between requests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **File Not Found**: Verify `Input.xlsx` and dictionary files exist in correct locations
3. **Network Errors**: Script includes retry logic, but some URLs may still fail due to server issues
4. **Encoding Issues**: Script handles various text encodings automatically

### Performance Notes

- Processing time depends on number of URLs and network speed
- Approximately 1-2 seconds per URL (including delay)
- Large articles may take longer to process

## Dependencies Explanation

- `pandas`: Excel/CSV file handling
- `requests`: HTTP requests for web scraping
- `beautifulsoup4`: HTML parsing and content extraction
- `openpyxl`: Excel file reading/writing
- `nltk`: Natural language processing and tokenization
- `textstat`: Additional text statistics (if needed)
- `lxml`: Fast XML/HTML parsing backend
- `urllib3`: HTTP client utilities
- `chardet`: Character encoding detection
