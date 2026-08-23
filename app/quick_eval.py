import sys, torch, numpy as np, random
sys.path.append('.')
from src.dataset import load_celebdf_samples, DeepfakeDataset, get_val_transforms
from src.model import build_model, load_checkpoint
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

samples = load_celebdf_samples('data/processed/celeb-df-v2')
random.seed(42)
random.shuffle(samples)
samples = samples[:2000]
print(f'Evaluating on {len(samples)} samples...')

loader = DataLoader(DeepfakeDataset(samples, get_val_transforms(299)),
                    batch_size=8, shuffle=False, num_workers=0)
model = build_model('xception', pretrained=False)
model = load_checkpoint(model, 'models/best_xception_celebdf.pth')
model.eval()

all_labels, all_probs = [], []
with torch.no_grad():
    for i, (imgs, labels) in enumerate(loader):
        probs = torch.sigmoid(model(imgs)).numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        if i % 10 == 0:
            print(f'  {len(all_labels)}/{len(samples)} done...')

labels = np.array(all_labels)
probs = np.array(all_probs)
print(f'AUC:      {roc_auc_score(labels, probs):.4f}')
print(f'Accuracy: {accuracy_score(labels, (probs>=0.5).astype(int)):.4f}')
print('DONE')  