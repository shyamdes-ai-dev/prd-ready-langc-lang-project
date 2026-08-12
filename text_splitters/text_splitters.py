"""
Text Splitting & Chunking Strategies
====================================
Demonstrates various text splitting techniques used to chunk documents for vector search & RAG:
1. `RecursiveCharacterTextSplitter`: Splits text hierarchically using a list of delimiters
   (`["\n\n", "\n", " ", ""]`), maintaining semantic cohesion across paragraphs and sentences.
2. Chunk Size & Overlap Comparison: Analyzes chunk count and overlap behavior across size thresholds.
3. `MarkdownHeaderTextSplitter`: Preserves markdown header hierarchies (`#`, `##`, `###`) and injects
   them as metadata tags into each chunk.
4. Language-Aware Code Splitter (`from_language(Language.PYTHON)`): Respects programming language
   AST boundaries (functions, classes, docstrings).
"""

from dotenv import load_dotenv
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Load environment configuration
load_dotenv()

# Sample Markdown document for splitting demonstrations
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

# Sample Python source code for AST-aware code splitting demonstrations
SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_splitter():
    """
    Demonstrates `RecursiveCharacterTextSplitter`.
    - `chunk_size`   : Maximum character length per chunk.
    - `chunk_overlap`: Number of overlapping characters shared between adjacent chunks
                       to prevent breaking key semantic context across chunk boundaries.
    - `separators`   : Delimiters tried in descending order of structural priority.
    """
    print("=== 1. Recursive Character Text Splitter ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text=SAMPLE_TEXT)

    print(f"Original Text Length : {len(SAMPLE_TEXT)} characters")
    print(f"Total Chunks Created : {len(chunks)}")
    print(f"Chunk Sizes (chars)  : {[len(c) for c in chunks]}")
    print(f"Chunk 1 Preview:\n{chunks[0][:120]}...\n")


def chunk_size_comparison():
    """
    Compares the total number of chunks created when varying chunk_size and chunk_overlap.
    Helps developers balance embedding granularity against retrieval context window.
    """
    print("=== 2. Chunk Size & Overlap Comparison ===")
    chunk_sizes = [100, 200, 500, 1000, 2000]

    for chunk_size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_size // 5
        )
        chunks = splitter.split_text(text=SAMPLE_TEXT)
        print(f"Chunk Size: {chunk_size:4d} | Overlap: {chunk_size//5:3d} -> Resulting Chunks: {len(chunks)}")
    print()


def markdown_splitters():
    """
    Demonstrates `MarkdownHeaderTextSplitter`.
    Splits text along markdown headings and attaches the heading hierarchy
    (H1, H2, H3) as metadata attributes on each returned Document object.
    """
    print("=== 3. Markdown Header Aware Text Splitter ===")
    headers_to_consider = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_consider, strip_headers=False
    )
    chunks = splitter.split_text(text=SAMPLE_TEXT)

    print(f"Extracted {len(chunks)} header-based chunks:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print("Metadata / Header Lineage:", chunk.metadata)
        print("Page Content Preview     :\n", chunk.page_content[:100].strip(), "...")
    print()


def code_splitter():
    """
    Demonstrates language-specific splitting via `RecursiveCharacterTextSplitter.from_language()`.
    Uses grammar rules specific to Python to avoid breaking function definitions or docstrings mid-sentence.
    """
    print("=== 4. Python AST-Aware Code Splitter ===")
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=300, chunk_overlap=30
    )
    chunks = python_splitter.split_text(text=SAMPLE_CODE)

    print(f"Split Python code into {len(chunks)} logical chunk(s):")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Code Chunk {i} ---")
        print(chunk.strip())
        print("-" * 40)
    print()


def main():
    """Execute all text splitter demonstrations."""
    recursive_splitter()
    chunk_size_comparison()
    markdown_splitters()
    code_splitter()


if __name__ == "__main__":
    main()
