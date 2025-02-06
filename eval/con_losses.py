def contrastive_loss(q, k, queue, temperature=0.07):
    # Positive logits
    pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    # Negative logits
    neg = torch.einsum('nc,ck->nk', [q, queue.T])
    # Logits and labels
    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(q.device)
    # Cross-entropy loss
    return nn.CrossEntropyLoss()(logits, labels)