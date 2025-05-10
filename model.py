import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model='paraphrase-MiniLM-L6-v2'):
        super().__init__()
        self.embedder = SentenceTransformer(embedding_model)
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
    def forward_once(self, x):
        if isinstance(x, str):
            x = [x]
        emb = self.embedder.encode(x, convert_to_tensor=True)
        return self.projection(emb)
    def forward(self, anchor, positive, negative):
        a = self.forward_once(anchor)
        p = self.forward_once(positive)
        n = self.forward_once(negative)
        return a, p, n
class QASystem:
    def __init__(self, model_path=None):
        self.model = SiameseNetwork()
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
                print("Loaded trained model weights")
            except FileNotFoundError:
                print("No trained model found, using base embeddings")
        self.model.eval()
        self.qa_pairs = []
        self.question_embeddings = []
    def load_data(self, qa_pairs):
        self.qa_pairs = qa_pairs
        questions = [q["question"] for q in qa_pairs]
        with torch.no_grad():
            self.question_embeddings = self.model.forward_once(questions)
    def find_answer(self, question, threshold=0.7):
        with torch.no_grad():
            q_emb = self.model.forward_once(question)
            sims = F.cosine_similarity(q_emb, self.question_embeddings)
            max_idx = torch.argmax(sims).item()
            return (
                self.qa_pairs[max_idx]["answer"] 
                if sims[max_idx] > threshold 
                else "Sorry, I don't have information about that."
            )