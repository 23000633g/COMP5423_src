# GloVe Dataset Chatbot

This project is a chatbot trained on the Titanic dataset. It can answer various questions about the passengers and other details from the dataset.

## Features

- Answer questions about the Titanic dataset.
- Evaluate responses using BLEU, ROUGE, METEOR, and BERTScore metrics.
- Utilize a vector store for name retrieval.

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/COMP5423_src.git
    cd COMP5423_src
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download necessary NLTK data:**
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    ```

4. **Prepare the dataset:**
    Ensure the `titanic.csv` file is in the project directory.

## Usage

### Running the Chatbot

To start the chatbot interface, run:

```sh
python main.py