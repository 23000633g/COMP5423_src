import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from DataFrameAgent import DataFrameAgent
import pandas as pd

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Define evaluation metrics
def evaluate_bleu(reference, candidate):
    if isinstance(reference, str) and isinstance(candidate, str):
        return sentence_bleu([reference.split()], candidate.split())
    return 0

def evaluate_rouge(reference, candidate):
    if isinstance(reference, str) and isinstance(candidate, str):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores
    return {}

def evaluate_meteor(reference, candidate):
    if isinstance(reference, str) and isinstance(candidate, str):
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        return meteor_score([reference_tokens], candidate_tokens)
    return 0

def evaluate_bertscore(references, candidates):
    references = [ref if isinstance(ref, str) else "" for ref in references]
    candidates = [cand if isinstance(cand, str) else "" for cand in candidates]
    P, R, F1 = bert_score(candidates, references, lang='en', rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

# Prepare evaluation data
evaluation_data = [
    {"question": "Who is the oldest passenger?", "answer": "he oldest passenger is Mr. Algernon Henry Wilson Barkworth, who is 80 years old."},
    {"question": "How many passengers survived?", "answer": "Total of 342 passengers survived."},
    {"question": "Who is the youngest passager?", "answer": "The youngest passenger is Thomas, Master. Assad Alexander."},
    {"question": "What is the average age of the passagers?", "answer": "The average age of the passagers is 29.7 years old."},
    {"question": "What is the average fare paid by the passagers?", "answer": "The average fare paid by the passagers is $32.20."},
    {"question": "What is the average fare paid by the passagers in each class?", "answer": " Class 1: $84.15- Class 2: $20.66- Class 3: $13.68"},
    {"question": "What is the name of the passager who paid the most fare?", "answer": "Cardeza, Mr. Thomas Drake Martinez."},
    {"question": "What is the name of the passager who paid the most fare in each class?", "answer": "Class 1: Ward, Miss. Anna\n- Class 2: Hood, Mr. Ambrose Jr- Class 3: Sage, Master. Thomas Henry"},
    {"question": "What is the name of the passager who paid the least fare in each class?", "answer": "The name of the passenger who paid the least fare in each class is:- Class 1: Harrison, Mr. William- Class 2: Mr. Francis Frank - Class 3: Leonard, Mr. Lionel"},
    {"question": "What's the mean, mode, and median of female passengers' ages?", "answer": "The mean 27.92, the mode is 24, and the median is 27."},
]

# Initialize the DFagent
df = pd.read_csv("titanic.csv")
agent = DataFrameAgent(df)


# Simulate user interactions and collect responses
predicted_answers = []
true_answers = [data["answer"] for data in evaluation_data]
for data in evaluation_data:
    response = agent.ask(data["question"])
    predicted_answers.append(response['output'])

print("Predicted Answers: ", predicted_answers)
# Evaluate the system's performance
bleu_scores = [evaluate_bleu(true, pred) for true, pred in zip(true_answers, predicted_answers)]
rouge_scores = [evaluate_rouge(true, pred) for true, pred in zip(true_answers, predicted_answers)]
meteor_scores = [evaluate_meteor(true, pred) for true, pred in zip(true_answers, predicted_answers)]
bertscore_P, bertscore_R, bertscore_F1 = evaluate_bertscore(true_answers, predicted_answers)

# Print evaluation results
print(f"BLEU Scores: {bleu_scores}")
print(f"Avg BLEU Score: {sum(bleu_scores) / len(bleu_scores)}")
# print(f"ROUGE Scores: {rouge_scores}")
print(f"METEOR Scores: {meteor_scores}")
print(f"Avg METEOR Score: {sum(meteor_scores) / len(meteor_scores)}")
print(f"BERTScore - Precision: {bertscore_P}, Recall: {bertscore_R}, F1: {bertscore_F1}")

