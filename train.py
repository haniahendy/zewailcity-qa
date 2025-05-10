import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SiameseNetwork
from data import qa_pairs, generate_triplets

def train():
    if not qa_pairs:
        raise ValueError("No Q&A pairs found in data.py")
    try:
        triplets = generate_triplets()
        if not triplets:
            raise ValueError("No triplets generated - check your data")
    except Exception as e:
        print(f"Error generating triplets: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training with {len(triplets)} triplets")
    model = SiameseNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for i in range(0, len(triplets), 8):
            batch = triplets[i:i+8]
            anchors = [t[0] for t in batch]
            positives = [t[1] for t in batch]
            negatives = [t[2] for t in batch]
            optimizer.zero_grad()
            try:
                a, p, n = model(anchors, positives, negatives)
                loss = criterion(a, p, n)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error in batch {i//8}: {e}")
                continue
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(triplets)//8):.4f}")
    torch.save(model.state_dict(), "siamese_model.pth")
    print("Training complete. Model saved to siamese_model.pth")
if __name__ == "__main__":
    train()