from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)

from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Sample documents for testing
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text=SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"chunk sizes: {[len(c) for c in chunks]}")
    print(f"Chunk preview: {chunks[0][:100]}...")

def chunk_size_comparison():
    chunk_sizes = [100,200,500,1000,2000]

    for chunk_size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_size //5)
        chunks = splitter.split_text(text=SAMPLE_TEXT)
        print(f" Size {chunk_size}: {len(chunks)} chunks")
 

def markdown_splitters():
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]



    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider,
                                        strip_headers=False)
    chunks = splitter.split_text(text=SAMPLE_TEXT)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk.page_content)
        print("Headers: ",chunk)
        print("-" * 40)


def code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=500, chunk_overlap=50)
    chunks = python_splitter.split_text(text=SAMPLE_CODE)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 40)

code_splitter()
