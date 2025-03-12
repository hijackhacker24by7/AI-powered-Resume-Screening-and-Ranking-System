import os
import string
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download necessary NLTK data files (only needed once)
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return text

def load_resumes(directory):
    """
    Loads resumes from a given directory.
    Supports .txt and .pdf files.
    Returns a dictionary in the format: {filename: text_content}
    """
    resumes = {}
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if filename.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(full_path)
            if resume_text.strip():
                resumes[filename] = resume_text
            else:
                print(f"Warning: {filename} may be empty or not extractable.")
        elif filename.lower().endswith('.txt'):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    resumes[filename] = f.read()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Skipping unsupported file format: {filename}")
    return resumes

def preprocess_text(text):
    """
    Preprocess the text data:
    - Converts to lowercase.
    - Removes punctuation.
    - (Further tokenization/lemmatization steps can be added if desired.)
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def screen_and_rank(job_description, resumes):
    """
    Ranks resumes based on cosine similarity computed between
    the job description and each resume's processed text.
    Returns a list of tuples: (filename, similarity_score)
    """
    # Preprocess the job description and resume texts
    job_description = preprocess_text(job_description)
    resume_texts = [preprocess_text(text) for text in resumes.values()]
    
    # Combine texts into a corpus; first element is the job description
    corpus = [job_description] + resume_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # The first vector is for the job description
    job_vec = tfidf_matrix[0]
    resume_vecs = tfidf_matrix[1:]
    
    # Compute cosine similarity between the job description and each resume
    similarities = cosine_similarity(job_vec, resume_vecs).flatten()
    
    # Pair each resume name with its corresponding similarity score
    resume_names = list(resumes.keys())
    resume_scores = list(zip(resume_names, similarities))
    
    # Sort resumes based on similarity (descending order)
    ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)
    return ranked_resumes

if __name__ == '__main__':
    # Define a sample job description
    job_description = (
        "We are seeking a data science professional with expertise in Python, machine learning, "
        "and analytical skills. Experience with NLP, resume screening techniques, and handling unstructured data is a plus."
    )
    
    # Load resumes from the directory (resumes should be placed in a folder named 'resumes')
    resumes = load_resumes('resumes')
    
    if not resumes:
        print("No resumes found in the directory.")
    else:
        # Obtain a ranked list of resumes
        ranked_results = screen_and_rank(job_description, resumes)
        
        # Display the ranked resumes along with similarity scores
        print("Ranked Resumes:")
        for filename, score in ranked_results:
            print(f"{filename}: {score:.3f}")
