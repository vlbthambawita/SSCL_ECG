import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
        super(MoCo, self).__init__()
        self.query_encoder = encoder(num_classes=dim)  # Query encoder
        self.key_encoder = encoder(num_classes=dim)   # Key encoder

        # Initialize key encoder weights as query encoder weights
        self.key_encoder.load_state_dict(self.query_encoder.state_dict())
        for param in self.key_encoder.parameters():
            param.requires_grad = False  # Key encoder is not updated via gradients

        # Queue for storing keys
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.momentum = momentum
        self.temperature = temperature

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        # Update key encoder using momentum
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        # Update the queue with new keys
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue.shape[1]  # Circular pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # Compute query features
        q = self.query_encoder(im_q)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = self.key_encoder(im_k)
            k = F.normalize(k, dim=1)

        # Contrastive loss computation
        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # Logits and labels
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # Update the queue
        self.dequeue_and_enqueue(k)

        return F.cross_entropy(logits, labels)


@torch.no_grad()
def momentum_update_key_encoder(query_encoder, key_encoder, momentum=0.999):
    for param_q, param_k in zip(query_encoder.parameters(), key_encoder.parameters()):
        param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data


class MemoryQueue:
    def __init__(self, feature_dim=128, queue_size=65536):
        self.queue = torch.randn(queue_size, feature_dim)
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.queue_ptr = 0
        self.queue_size = queue_size

    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = self.queue_ptr % self.queue_size
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_ptr += batch_size

def train(self):

    for batch in dataloader:
        # Get augmented pairs
        x_q, x_k = augment(batch), augment(batch)
        # Compute query and key features
        q = query_encoder(x_q)
        with torch.no_grad():
            k = key_encoder(x_k)
        # Contrastive loss
        loss = contrastive_loss(q, k, memory_queue.queue)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Momentum update and enqueue
        momentum_update_key_encoder(query_encoder, key_encoder)
        memory_queue.dequeue_and_enqueue(k)