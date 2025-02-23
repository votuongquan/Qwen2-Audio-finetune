import torch
def compute_acc(logits,labels):
    _,labels_len = labels.shape
    preds = torch.argmax(logits,dim=-1)
    labels_indices = labels != -100 
    acc = torch.sum(preds[:,-labels_len-1:-1][labels_indices] == labels[labels_indices]).float() /torch.sum(labels_indices).float()
    return acc