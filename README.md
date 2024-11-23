# GloVe Dataset Chatbot

GloVe Dataset Chatbot is an interactive chatbot designed to answer questions about the Titanic dataset. The chatbot uses various natural language processing tools and models to provide accurate and helpful responses to user queries.

## Table of Contents

- Installation
- Usage
- [Project Structure](#project-structure)
- Evaluation
- Contributing
- License

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/glove-dataset-chatbot.git
    cd glove-dataset-chatbot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the necessary NLTK data:
    ```sh
    python -m nltk.downloader wordnet omw-1.4 punkt
    ```

## Usage

1. Start the chatbot interface:
    ```sh
    python main.py
    ```

2. Open your web browser and navigate to the URL provided by Gradio to interact with the chatbot.

## Project Structure

```
.
├── .gitignore
├── ChatBotAgent.py
├── DataFrameAgent.py
├── evaluation.py
├── main.py
├── NounVectorStore.py
├── README.md
├── requirements.txt
└── utils.py
```

- **ChatBotAgent.py**: Contains the ChatBotAgent class which handles user queries and interacts with the DataFrameAgent and NounVectorStore.
- **DataFrameAgent.py**: Contains the DataFrameAgent class which processes queries related to the Titanic dataset.
- **evaluation.py**: Contains functions to evaluate the chatbot's performance using various metrics like BLEU, ROUGE, METEOR, and BERTScore.
- **main.py**: Launches the Gradio interface for the chatbot.
- **NounVectorStore.py**: Contains the NounVectorStore class which helps in retrieving proper nouns from the dataset.
- **utils.py**: Contains utility functions for getting chat and embedding models.
- **requirements.txt**: Lists the dependencies required for the project.

## Evaluation

To evaluate the chatbot's performance, run the evaluation.py 
script:
```sh
python evaluation.py
```
This script will print the evaluation results using BLEU, ROUGE, METEOR, and BERTScore metrics.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.