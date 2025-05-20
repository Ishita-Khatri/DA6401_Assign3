# %%
def load_dakshina_lexicon_pairs(filepath):
    pairs=[]
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            parts = line.split('\t')
            if len(parts) != 3:
                continue  # skip malformed lines
            devanagari_word, latin_word,_ = parts
            pairs.append((latin_word, devanagari_word))  # reverse order
    return pairs

# %%
filepath = "/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
pairs = load_dakshina_lexicon_pairs(filepath)

for i in range(5):
    print(pairs[i])

# %%
def build_vocab(pairs, add_special_tokens=True):
    input_chars = set() # to ensure no repeated characters
    output_chars = set()

    # Collect unique characters from Latin (input) and Devanagari (output)
    for latin_word, devnagari_word in pairs:
        input_chars.update(list(latin_word))
        output_chars.update(list(devnagari_word))

    # Sort to keep it consistent
    input_chars = sorted(list(input_chars))
    output_chars = sorted(list(output_chars))

    # Add special tokens
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>'] if add_special_tokens else []

    input_vocab = special_tokens + input_chars
    output_vocab = special_tokens + output_chars

    # Create dictionaries
    input_char2idx = {ch: idx for idx, ch in enumerate(input_vocab)}
    input_idx2char = {idx: ch for ch, idx in input_char2idx.items()}

    output_char2idx = {ch: idx for idx, ch in enumerate(output_vocab)}
    output_idx2char = {idx: ch for ch, idx in output_char2idx.items()}

    return input_char2idx, input_idx2char, output_char2idx, output_idx2char

# %%
input_char2idx, input_idx2char, output_char2idx, output_idx2char = build_vocab(pairs)

print("Latin char2idx:", list(input_char2idx.items())[:5])
print("Devanagari idx2char:", list(output_idx2char.items())[:5])

print(len(list(output_char2idx.keys())))

# %%
import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, num_encoder_layers=1, cell_type='lstm', dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_size, hidden_size, num_encoder_layers,
                batch_first=True, dropout=dropout if num_encoder_layers > 1 else 0
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embed_size, hidden_size, num_encoder_layers,
                batch_first=True, dropout=dropout if num_encoder_layers > 1 else 0
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                embed_size, hidden_size, num_encoder_layers,
                batch_first=True, dropout=dropout if num_encoder_layers > 1 else 0
            )
    
    def forward(self, input_seq, lengths):
        """
        Forward pass for encoder
        
        Args:
            input_seq: Input sequence tensor [batch_size, max_seq_len]
            lengths: Actual lengths of input sequences (tensor)
            
        Returns:
            None: Instead of encoder outputs (to avoid DataParallel issues)
            hidden: Hidden state for decoder initialization
        """
        batch_size = input_seq.size(0)
        
        # Important: ensure lengths is on CPU before using it
        if lengths.is_cuda:
            lengths = lengths.cpu()
        
        # Convert input to embeddings
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, embed_size]
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Process with RNN
        if self.cell_type == 'lstm':
            # Don't return outputs to avoid DataParallel gathering issues
            outputs, (hidden, cell) = self.rnn(packed)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.rnn(packed)
            return outputs, hidden


# %%
class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size, num_decoder_layers=1, cell_type='lstm', dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.output_vocab_size = output_vocab_size
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_vocab_size, embed_size)
        
        # RNN layer
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_size, hidden_size, num_decoder_layers,
                batch_first=True, dropout=dropout if num_decoder_layers > 1 else 0
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embed_size, hidden_size, num_decoder_layers,
                batch_first=True, dropout=dropout if num_decoder_layers > 1 else 0
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                embed_size, hidden_size, num_decoder_layers,
                batch_first=True, dropout=dropout if num_decoder_layers > 1 else 0
            )
        
        # Output layer
        self.out = nn.Linear(hidden_size, output_vocab_size)
    
    def _convert_encoder_hidden(self, encoder_hidden):
        """
        Convert encoder hidden state to fit decoder dimensions
        
        Args:
            encoder_hidden: Encoder's hidden state
                           For RNN/GRU: tensor of shape (num_encoder_layers, batch_size, hidden_size)
                           For LSTM: tuple of two tensors with that shape
        
        Returns:
            Hidden state with shape compatible with decoder
        """
        if self.cell_type == 'lstm':
            # For LSTM, encoder_hidden is a tuple (hidden, cell)
            hidden, cell = encoder_hidden
            
            # Get shapes
            num_encoder_layers, batch_size, hidden_size = hidden.shape
            
            # Return as is if dimensions already match
            if num_encoder_layers == self.num_decoder_layers:
                return encoder_hidden
            
            # Initialize decoder hidden state
            decoder_hidden = torch.zeros(self.num_decoder_layers, batch_size, hidden_size, device=hidden.device)
            decoder_cell = torch.zeros(self.num_decoder_layers, batch_size, hidden_size, device=cell.device)
            
            # Fill decoder hidden state
            if num_encoder_layers >= self.num_decoder_layers:
                # Take the last layers from encoder
                decoder_hidden = hidden[-self.num_decoder_layers:]
                decoder_cell = cell[-self.num_decoder_layers:]
            else:
                # Copy all available layers from encoder
                decoder_hidden[:num_encoder_layers] = hidden
                decoder_cell[:num_encoder_layers] = cell
                
                # Fill remaining layers with the last encoder layer
                for i in range(num_encoder_layers, self.num_decoder_layers):
                    decoder_hidden[i] = hidden[-1]
                    decoder_cell[i] = cell[-1]
            
            return (decoder_hidden, decoder_cell)
        
        else:  # RNN or GRU
            # Get shapes
            num_encoder_layers, batch_size, hidden_size = encoder_hidden.shape
            
            # Return as is if dimensions already match
            if num_encoder_layers == self.num_decoder_layers:
                return encoder_hidden
            
            # Initialize decoder hidden state
            decoder_hidden = torch.zeros(self.num_decoder_layers, batch_size, hidden_size, device=encoder_hidden.device)
            
            # Fill decoder hidden state
            if num_encoder_layers >= self.num_decoder_layers:
                # Take the last layers from encoder
                decoder_hidden = encoder_hidden[-self.num_decoder_layers:]
            else:
                # Copy all available layers from encoder
                decoder_hidden[:num_encoder_layers] = encoder_hidden
                
                # Fill remaining layers with the last encoder layer
                for i in range(num_encoder_layers, self.num_decoder_layers):
                    decoder_hidden[i] = encoder_hidden[-1]
            
            return decoder_hidden
    
    def forward(self, input_seq, hidden):
        """
        Forward pass for decoder
        
        Args:
            input_seq: Input sequence tensor [batch_size, 1]
            hidden: Hidden state from encoder or previous decoder step
                   Will be automatically converted to match decoder dimensions
        
        Returns:
            output: Output logits
            hidden: Updated hidden state
        """
        # Convert encoder hidden state if this is the first decoder step
        hidden = self._convert_encoder_hidden(hidden)
        
        # Convert input to embeddings
        embedded = self.embedding(input_seq)  # [batch_size, 1, embed_size]
        
        # Process with RNN
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            output = self.out(output)  # [batch_size, 1, output_vocab_size]
            return output, (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            output = self.out(output)  # [batch_size, 1, output_vocab_size]
            return output, hidden

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TransliterationDataset(Dataset):
    def __init__(self, pairs, input_char2idx, output_char2idx):
        
        '''
        pairs: list of (latin_word, devnagari_word) tuples.
        input_char2idx: dictionary mapping each Latin character to an index.
        output_char2idx: dictionary mapping each Devanagari character to an index.
        '''
            
        self.pairs = pairs
        self.input_char2idx = input_char2idx
        self.output_char2idx = output_char2idx

    # This converts a word into a list of token indices, e.g., India -> [8,13,3,8,0]
    def encode_word(self, word, char2idx, add_sos_eos=False):
        tokens = [char2idx.get(c, char2idx['<unk>']) for c in word]
        if add_sos_eos:
            tokens = [char2idx['<sos>']] + tokens + [char2idx['<eos>']]
        return tokens

    #  Give the total number of latin, devnagri pairs in the dataset
    def __len__(self): 
        return len(self.pairs)

    # This takes the index of the word in latin and gets the latin, devnagri pair. 
        # Then, it converts each word to list of indices and gives the pair of list of indices
    def __getitem__(self, idx):
        latin, devnagari = self.pairs[idx]
        input_ids = self.encode_word(latin, self.input_char2idx)
        target_ids = self.encode_word(devnagari, self.output_char2idx, add_sos_eos=True)
        return input_ids, target_ids

# %%
#  Adds pad tokens, given the sequnece, maximum length and pad-token
def pad_seq(seq, max_len, pad_token):
    return seq + [pad_token] * (max_len - len(seq))

def collate_fn(batch):
    '''
    batch: List of tuples [(input1, target1), (input2, target2), ...]

    '''
    input_seqs, target_seqs = zip(*batch)

    input_max_len = max(len(seq) for seq in input_seqs)
    target_max_len = max(len(seq) for seq in target_seqs)

    # Adds padding for seqeuces so that sequence length = maximum sequence length in the batch. 
    # Now all sequenes in the batch are of same length 
    input_padded = [pad_seq(seq, input_max_len, pad_token=input_char2idx['<pad>']) for seq in input_seqs]
    target_padded = [pad_seq(seq, target_max_len, pad_token=output_char2idx['<pad>']) for seq in target_seqs]

    input_tensor = torch.tensor(input_padded, dtype=torch.long)
    target_tensor = torch.tensor(target_padded, dtype=torch.long)

    input_lengths = torch.tensor([len(seq) for seq in input_seqs])
    target_lengths = torch.tensor([len(seq) for seq in target_seqs])

    return input_tensor, input_lengths, target_tensor, target_lengths

# %%
sweep_config = {
    'method': 'bayes',  # Could also be 'random' or 'grid'
    'metric': {
        'name': 'token_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'embed_size': {'values': [16, 32, 64]},
        'num_encoder_layers': {'values': [1, 2, 3]},
        'num_decoder_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [16, 32, 64]},
        'cell_type': {'values': ['RNN', 'GRU', 'LSTM']},
        'dropout': {'values': [0.3, 0.4, 0.5]},
        'batch_size': {'values': [128, 256, 512]},
        'learning_rate': {'values': [5e-3, 1e-3, 5e-4]},
        'beam_size': {'values': [3, 4, 5]}
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 7
    }
}

# %%
import wandb

# %%
filepath_val = "/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
pairs_val = load_dakshina_lexicon_pairs(filepath_val)

dataset = TransliterationDataset(pairs, input_char2idx, output_char2idx)
dataset_val = TransliterationDataset(pairs_val, input_char2idx, output_char2idx)

# %%
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train():
    wandb.init()
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize encoder and decoder
    encoder = Encoder(
        input_vocab_size=len(input_char2idx),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_encoder_layers=config.num_encoder_layers,
        cell_type=config.cell_type,
        dropout=config.dropout
    ).to(device)

    decoder = Decoder(
        output_vocab_size=len(output_char2idx),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_decoder_layers=config.num_decoder_layers,
        cell_type=config.cell_type,
        dropout=config.dropout
    ).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=output_char2idx['<pad>'])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    # Load validation set
    filepath_val = "/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    pairs_val = load_dakshina_lexicon_pairs(filepath_val)
    dataset_val = TransliterationDataset(pairs_val, input_char2idx, output_char2idx)

    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

    num_epochs = 20
    for epoch in range(num_epochs):
        # ======== TRAINING ========
        encoder.train()
        decoder.train()
        total_loss = 0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for input_tensor, input_lengths, target_tensor, target_lengths in pbar:
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
                decoder_input = target_tensor[:, 0].unsqueeze(1)  # <sos>
                decoder_hidden = encoder_hidden

                loss = 0
                max_target_len = target_tensor.size(1)

                for t in range(1, max_target_len):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    output = decoder_output.squeeze(1)
                    # print(f'output = {output}')
                    # print(f'target tensor = {target_tensor[:,t]}')
                    loss += criterion(output, target_tensor[:, t])
                    decoder_input = target_tensor[:, t].unsqueeze(1)  # Teacher forcing

                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

                encoder_optimizer.step()
                decoder_optimizer.step()

                total_loss += loss.item() / (max_target_len - 1)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {avg_loss:.4f}")

        # ======== VALIDATION ========
        encoder.eval()
        decoder.eval()
        correct_sequences = 0
        total_sequences = 0
        correct_tokens = 0
        total_tokens = 0
        beam_width = config.beam_size  # You can change this
        
        with torch.no_grad():
            for input_tensor, input_lengths, target_tensor, target_lengths in dataloader_val:
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)
        
                encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
                max_target_len = target_tensor.size(1)
                total_sequences += 1
        
                # Beam is a list of tuples: (sequence_so_far, cumulative_log_prob, decoder_hidden)
                beam = [([output_char2idx['<sos>']], 0.0, encoder_hidden)]
        
                completed_sequences = []
        
                for _ in range(1, max_target_len):
                    new_beam = []
                    for seq, score, hidden in beam:
                        decoder_input = torch.tensor([[seq[-1]]], device=device)
                        decoder_output, hidden_next = decoder(decoder_input, hidden)
                        log_probs = F.log_softmax(decoder_output.squeeze(1), dim=1)
        
                        topk_log_probs, topk_indices = log_probs.topk(beam_width)
        
                        for k in range(beam_width):
                            next_token = topk_indices[0][k].item()
                            next_score = score + topk_log_probs[0][k].item()
                            new_seq = seq + [next_token]
                            new_beam.append((new_seq, next_score, hidden_next))
        
                    # Keep top `beam_width` beams with highest scores
                    beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        
                    # Move completed sequences out
                    beam, completed = [], []
                    for seq, score, hidden in new_beam:
                        if seq[-1] == output_char2idx['<eos>']:
                            completed_sequences.append((seq, score))
                        else:
                            beam.append((seq, score, hidden))
                    beam = sorted(beam, key=lambda x: x[1], reverse=True)[:beam_width]
        
                # Choose best completed or best incomplete beam
                if completed_sequences:
                    best_seq = max(completed_sequences, key=lambda x: x[1])[0]
                else:
                    best_seq = max(beam, key=lambda x: x[1])[0]
        
                # Remove <sos> if present
                if best_seq[0] == output_char2idx['<sos>']:
                    best_seq = best_seq[1:]
        
                # Compare prediction with target
                target_seq = target_tensor[0, 1:].tolist()
                pad_idx = output_char2idx['<pad>']
        
                # Token accuracy
                for pred_token, tgt_token in zip(best_seq, target_seq):
                    if tgt_token == pad_idx:
                        break
                    if pred_token == tgt_token:
                        correct_tokens += 1
                    total_tokens += 1
        
                # Sequence accuracy
                target_trimmed = [t for t in target_seq if t != pad_idx]
                best_seq_trimmed = best_seq[:len(target_trimmed)]
                if best_seq_trimmed == target_trimmed:
                    correct_sequences += 1
        
                # Optional print
                # predicted_word = indices_to_words([best_seq], output_idx2char)[0]
                # actual_word = indices_to_words([target_trimmed], output_idx2char)[0]
                # # print(f"Predicted: {predicted_word.ljust(20)} | Actual: {actual_word}")
        
        sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        
        print(f"Token Accuracy: {token_accuracy:.4f}")
        print(f"Sequence Accuracy: {sequence_accuracy:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "token_accuracy": token_accuracy,
            "sequence_accuracy": sequence_accuracy
        })

# %%
sweep_id = wandb.sweep(sweep_config, project="DA6401 Assign3")
wandb.agent(sweep_id, function=train, count=20)
wandb.finish()

# %%
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

filepath_test = "/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
pairs_test = load_dakshina_lexicon_pairs(filepath_test)
dataset_test = TransliterationDataset(pairs_test, input_char2idx, output_char2idx)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

filepath_val = "/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
pairs_val = load_dakshina_lexicon_pairs(filepath_val)
dataset_val = TransliterationDataset(pairs_val, input_char2idx, output_char2idx)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

dataset = TransliterationDataset(pairs, input_char2idx, output_char2idx)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#  Best configuration
embed_size=64
num_encoder_layers=3
num_decoder_layers=3
hidden_size=64
cell_type='lstm'
dropout=0.4
batch_size=128
learning_rate=0.005
beam_size=4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize encoder and decoder
encoder = Encoder(
    input_vocab_size=len(input_char2idx),
    embed_size=embed_size,
    hidden_size=hidden_size,
    num_encoder_layers=num_encoder_layers,
    cell_type=cell_type,
    dropout=dropout
).to(device)

decoder = Decoder(
    output_vocab_size=len(output_char2idx),
    embed_size=embed_size,
    hidden_size=hidden_size,
    num_decoder_layers=num_decoder_layers,
    cell_type=cell_type,
    dropout=dropout
).to(device)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=output_char2idx['<pad>'])

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

num_epochs = 20
for epoch in range(num_epochs):
    # ======== TRAINING ========
    encoder.train()
    decoder.train()
    total_loss = 0

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for input_tensor, input_lengths, target_tensor, target_lengths in pbar:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
            decoder_input = target_tensor[:, 0].unsqueeze(1)  # <sos>
            decoder_hidden = encoder_hidden

            loss = 0
            max_target_len = target_tensor.size(1)

            for t in range(1, max_target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                output = decoder_output.squeeze(1)
                # print(f'output = {output}')
                # print(f'target tensor = {target_tensor[:,t]}')
                loss += criterion(output, target_tensor[:, t])
                decoder_input = target_tensor[:, t].unsqueeze(1)  # Teacher forcing

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item() / (max_target_len - 1)

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {avg_loss:.4f}")

    # ======== VALIDATION ========
    encoder.eval()
    decoder.eval()
    correct_sequences = 0
    total_sequences = 0
    correct_tokens = 0
    total_tokens = 0
    beam_width = beam_size  # You can change this
    with torch.no_grad():
        for input_tensor, input_lengths, target_tensor, target_lengths in dataloader_val:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
    
            encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
            max_target_len = target_tensor.size(1)
            total_sequences += 1
    
            # Beam is a list of tuples: (sequence_so_far, cumulative_log_prob, decoder_hidden)
            beam = [([output_char2idx['<sos>']], 0.0, encoder_hidden)]
    
            completed_sequences = []
    
            for _ in range(1, max_target_len):
                new_beam = []
                for seq, score, hidden in beam:
                    decoder_input = torch.tensor([[seq[-1]]], device=device)
                    decoder_output, hidden_next = decoder(decoder_input, hidden)
                    log_probs = F.log_softmax(decoder_output.squeeze(1), dim=1)
    
                    topk_log_probs, topk_indices = log_probs.topk(beam_width)
    
                    for k in range(beam_width):
                        next_token = topk_indices[0][k].item()
                        next_score = score + topk_log_probs[0][k].item()
                        new_seq = seq + [next_token]
                        new_beam.append((new_seq, next_score, hidden_next))
    
                # Keep top `beam_width` beams with highest scores
                beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
    
                # Move completed sequences out
                beam, completed = [], []
                for seq, score, hidden in new_beam:
                    if seq[-1] == output_char2idx['<eos>']:
                        completed_sequences.append((seq, score))
                    else:
                        beam.append((seq, score, hidden))
                beam = sorted(beam, key=lambda x: x[1], reverse=True)[:beam_width]
    
            # Choose best completed or best incomplete beam
            if completed_sequences:
                best_seq = max(completed_sequences, key=lambda x: x[1])[0]
            else:
                best_seq = max(beam, key=lambda x: x[1])[0]
    
            # Remove <sos> if present
            if best_seq[0] == output_char2idx['<sos>']:
                best_seq = best_seq[1:]
    
            # Compare prediction with target
            target_seq = target_tensor[0, 1:].tolist()
            pad_idx = output_char2idx['<pad>']
    
            # Token accuracy
            for pred_token, tgt_token in zip(best_seq, target_seq):
                if tgt_token == pad_idx:
                    break
                if pred_token == tgt_token:
                    correct_tokens += 1
                total_tokens += 1
    
            # Sequence accuracy
            target_trimmed = [t for t in target_seq if t != pad_idx]
            best_seq_trimmed = best_seq[:len(target_trimmed)]
            if best_seq_trimmed == target_trimmed:
                correct_sequences += 1
    
            # Optional print
            # predicted_word = indices_to_words([best_seq], output_idx2char)[0]
            # actual_word = indices_to_words([target_trimmed], output_idx2char)[0]
            # # print(f"Predicted: {predicted_word.ljust(20)} | Actual: {actual_word}")
    
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    print(f"Token Accuracy: {token_accuracy:.4f}")
    print(f"Sequence Accuracy: {sequence_accuracy:.4f}")

# %%
# ======== TEST ========
encoder.eval()
decoder.eval()
correct_sequences = 0
total_sequences = 0
correct_tokens = 0
total_tokens = 0
beam_width = beam_size  # You can change this
result=[]
with torch.no_grad():
    for input_tensor, input_lengths, target_tensor, target_lengths in dataloader_test:
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
        max_target_len = target_tensor.size(1)
        total_sequences += 1

        # Beam is a list of tuples: (sequence_so_far, cumulative_log_prob, decoder_hidden)
        beam = [([output_char2idx['<sos>']], 0.0, encoder_hidden)]

        completed_sequences = []

        for _ in range(1, max_target_len):
            new_beam = []
            for seq, score, hidden in beam:
                decoder_input = torch.tensor([[seq[-1]]], device=device)
                decoder_output, hidden_next = decoder(decoder_input, hidden)
                log_probs = F.log_softmax(decoder_output.squeeze(1), dim=1)

                topk_log_probs, topk_indices = log_probs.topk(beam_width)

                for k in range(beam_width):
                    next_token = topk_indices[0][k].item()
                    next_score = score + topk_log_probs[0][k].item()
                    new_seq = seq + [next_token]
                    new_beam.append((new_seq, next_score, hidden_next))

            # Keep top `beam_width` beams with highest scores
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

            # Move completed sequences out
            beam, completed = [], []
            for seq, score, hidden in new_beam:
                if seq[-1] == output_char2idx['<eos>']:
                    completed_sequences.append((seq, score))
                else:
                    beam.append((seq, score, hidden))
            beam = sorted(beam, key=lambda x: x[1], reverse=True)[:beam_width]

        # Choose best completed or best incomplete beam
        if completed_sequences:
            best_seq = max(completed_sequences, key=lambda x: x[1])[0]
        else:
            best_seq = max(beam, key=lambda x: x[1])[0]

        # Remove <sos> if present
        if best_seq[0] == output_char2idx['<sos>']:
            best_seq = best_seq[1:]
        
        # Compare prediction with target
        target_seq = target_tensor[0, 1:].tolist()
        pad_idx = output_char2idx['<pad>']

        # Token accuracy
        for pred_token, tgt_token in zip(best_seq, target_seq):
            if tgt_token == pad_idx:
                break
            if pred_token == tgt_token:
                correct_tokens += 1
            total_tokens += 1

        # Sequence accuracy
        target_trimmed = [t for t in target_seq if t != pad_idx]
        best_seq_trimmed = best_seq[:len(target_trimmed)]
        if best_seq_trimmed == target_trimmed:
            correct_sequences += 1
        
        if best_seq[-1] == output_char2idx['<eos>']:
            best_seq = best_seq[:best_seq.index(output_char2idx['<eos>'])]
        predicted_word=''.join(output_idx2char[i] for i in best_seq)
        target_seq = target_tensor.tolist() if hasattr(target_tensor, 'tolist') else target_tensor
        if isinstance(target_seq[0], list):
            target_seq = target_seq[0]
        # Remove <sos> and truncate at <eos> if present
        if target_seq[0] == output_char2idx['<sos>']:
            target_seq = target_seq[1:]
        if output_char2idx.get('<eos>') in target_seq:
            target_seq = target_seq[:target_seq.index(output_char2idx['<eos>'])]
        
        target_word = ''.join(output_idx2char[i] for i in target_seq)
        
        input_seq = input_tensor.tolist() if hasattr(input_tensor, 'tolist') else input_tensor
        if isinstance(input_seq[0], list):
            input_seq = input_seq[0]

        if input_seq[0] == input_char2idx['<sos>']:
            input_seq = input_seq[1:]
        if input_char2idx.get('<eos>') in input_seq:
            input_seq = input_seq[:input_seq.index(input_char2idx['<eos>'])]
        input_word = ''.join(input_idx2char[i] for i in input_seq)

        result.append((input_word, predicted_word, target_word))
        # Optional print
        # predicted_word = indices_to_words([best_seq], output_idx2char)[0]
        # actual_word = indices_to_words([target_trimmed], output_idx2char)[0]
        # # print(f"Predicted: {predicted_word.ljust(20)} | Actual: {actual_word}")

sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
print("Test:")
print(f"Token Accuracy: {token_accuracy:.4f}")
print(f"Sequence Accuracy: {sequence_accuracy:.4f}")

# %%
print(result[0])

# %%
import csv

with open('predictions_vanilla.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Input', 'Predicted', 'Target'])  # Header
    for t, pred, target in result:
        writer.writerow([t, pred, target])


