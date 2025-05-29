import torch
from torch.nn.utils.rnn import pad_sequence
#padding y quitar la dimension esa rara (buscar q es)
def collate_fn(batch):
    audios, labels = zip(*batch)
    audios = [a.squeeze(0) if a.dim() == 2 else a for a in audios]
    audios_padded = pad_sequence(audios, batch_first=True)  # (batch_size, max_len)
    labels = torch.tensor(labels, dtype=torch.long)
    return audios_padded, labels
