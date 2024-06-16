```python
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm, trange
```


```python
class StackedLSTMCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout=0.0):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i, c_i = layer(x, (h_0[i], c_0[i]))

            # x for next layer
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return last_h_c, h_c_list
```

##Define the Summarizer


```python
class sLSTM(nn.Module):
  
  def __init__(self, input_size, hidden_size, num_layers=2):
    """
    Scoring LSTM.
    Unfolds the video sequence and predicts a probability value for a frame to
    belong to the summary.
    """
    super().__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    self.out = nn.Sequential(
        nn.Linear(hidden_size * 2, 1), # The times 2 indicates that the lstm is bi-directional.
        nn.Sigmoid()
    )

  def forward(self, features, init_hidden=None):
    """
    Args:
      features: [seq_len, 1, 500] (Compressed GoogLeNet's pool5 features)
    Returns:
      scores: [seq_len, 1]
    """
    self.lstm.flatten_parameters()

    # [seq_len, 1, hidden_size * 2]
    features, (h_n, c_n) = self.lstm(features)

    # [seq_len, 1]
    scores = self.out(features.squeeze(1))

    return scores


class eLSTM(nn.Module):
  """
  Encoder LSTM Module
  """
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

  def forward(self, frame_features):
    """
    Args:
      frame_features: [seq_len, 1, hidden_size]
    Returns:
      output: [seq_len, 1, hidden_size]
      last_hidden:
        h_last: [num_layers=2, 1, hidden_size]
        c_last: [num_layers=2, 1, hidden_size]
    """
    self.lstm.flatten_parameters()
    output, (h_last, c_last) = self.lstm(frame_features)

    return output, (h_last, c_last)


class dLSTM(nn.Module):
  """Decoder LSTM"""
  def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
    super().__init__()
    
    self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
    self.out = nn.Linear(hidden_size, input_size)
    self.input_size = input_size
  
  def forward(self, seq_len, encoder_output, init_hidden):
    """
    Args:
      seq_len: (int)
      encoder_output: [seq_len, 1, hidden_size]
      init_hidden:
        h [num_layers=2, 1, hidden_size]
        c [num_layers=2, 1, hidden_size]
    Returns:
      out_features: [seq_len, 1, hidden_size]
    """
    batch_size = init_hidden[0].size(1)
    hidden_size = init_hidden[0].size(2)

    x = Variable(torch.zeros(batch_size, self.input_size)).cuda()
    h, c = init_hidden  # (h_0, c_0): last state of eLSTM

    out_features = []
    for i in range(seq_len):
      # last_h: [1, hidden_size] (h from last layer)
      # last_c: [1, hidden_size] (c from last layer)
      # h: [2=num_layers, 1, hidden_size] (h from all layers)
      # c: [2=num_layers, 1, hidden_size] (c from all layers)
      
      (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
      x = self.out(last_h)
      # print("In line 91:")
      # print(x.size())
      out_features.append(x)
      # list of seq_len '[1, hidden_size]-sized Variables'
    return out_features


class AE(nn.Module):
  """LSTM Autoencoder."""
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()
    self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
    self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

  def forward(self, features):
    """
    Args:
      features: [seq_len, 1, hidden_size]
    Returns:
      decoded_features: [seq_len, 1, hidden_size]
    """
    seq_len = features.size(0)

    #encoder_output: [seq_len, 1, hidden_size]
    # h and c: [num_layers, 1, hidden_size]
    encoder_output, (h, c) = self.e_lstm(features)
    print(encoder_output.size())
    # [seq_len, 1, hidden_size]
    decoded_features = self.d_lstm(seq_len, encoder_output, init_hidden=(h, c))
    

    # [seq_len, 1, hidden_size]
    # reverse
    decoded_features.reverse()
    decoded_features = torch.stack(decoded_features) # list to tensor
    print(decoded_features.size())
    return decoded_features


class Summarizer(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()
    self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
    self.auto_enc = AE(input_size, hidden_size, num_layers)

  def forward(self, image_features):
    """
    Args:
      image_features: [seq_len, 1, hidden_size]
    Returns:
      scores: [seq_len, 1]
      decoded_features: [seq_len, 1, hidden_size]
    """

    # Estimate the probabities
    # [seq_len, 1]
    scores = self.s_lstm(image_features)

    # [seq_len, 1, hidden_size]
    weighted_features = image_features * scores.view(-1, 1, 1)

    decoded_features = self.auto_enc(weighted_features)

    return scores, decoded_features

```


```python
a = torch.randn(10, 1, 20).cuda()
ae = AE(20, 5).cuda()
```


```python
a.size()
```




    torch.Size([10, 1, 20])




```python
a_enc, (h, c) = ae.e_lstm(a)
```


```python
h.size()
```




    torch.Size([2, 1, 5])




```python
a_dec = ae(a)
```

    torch.Size([10, 1, 5])
    torch.Size([10, 1, 20])
    


```python
a_dec.size()
```




    torch.Size([10, 1, 20])



##Define the Discriminator


```python
class cLSTM(nn.Module):
  
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

  def forward(self, features, init_hidden=None):
    """
    Args:
      features: [seq_len, 1, input_size]
    Returns:
      last_h: [1, hidden_size]
    """
    self.lstm.flatten_parameters()

    # output: [seq_len, batch_size, hidden_size * num_directions]
    # h_n, c_n: [num_layers * num_directions, batch_size, hidden_size]
    # print("In cLSTM, features size: {}".format(features.size()))
    output, (h_n, c_n) = self.lstm(features, init_hidden)

    # [batch_size, hidden_size]
    last_h = h_n[-1]

    return last_h


class Discriminator(nn.Module):
  
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()
    self.cLSTM = cLSTM(input_size, hidden_size, num_layers)
    self.out = nn.Sequential(
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()
    )
  
  def forward(self, features):
    """
    Args:
      features: [seq_len, 1, hidden_size]
    Returns:
      h: [1, hidden_size]
         Last h from top layer of the discriminator
      prob: [batch_size=1, 1]
         Probability to be an original feature from the CNN
    """

    # [1, hidden_size]
    h = self.cLSTM(features)

    # [1]
    # Squeeze because the out will have a size [1, 1, 1] and it cannot be used
    # for the loss calculation
    prob = self.out(h).squeeze()

    return h, prob
```

##Loading the data


```python
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

VIDEO_NAME = 'summe'

class VideoData(Dataset):

  def __init__(self, mode, split_index):
    # Need split_index?
    self.mode = mode
    self.name = VIDEO_NAME
    self.datasets = ['eccv16_dataset_summe_google_pool5.h5',
                     'eccv16_dataset_tvsum_google_pool5.h5']
    self.splits_filename = [self.name + '_splits.json']
    self.splits = []
    self.split_index = split_index
    temp = {}

    if 'summe' in self.splits_filename[0]:
      self.filename = self.datasets[0]
    elif 'tvsum' in self.splits_filename[0]:
      self.filename = self.datasets[1]
    self.video_data = h5py.File(self.filename, "r")

    with open(self.splits_filename[0]) as f:
      data = json.loads(f.read())
      for split in data:
        temp['train_keys'] = split['train_keys']
        temp['test_keys'] = split['test_keys']
        self.splits.append(temp.copy())
  
  def __len__(self):
    self.len = len(self.splits[0][self.mode+'_keys'])
    return self.len
  
  def __getitem__(self, index):
    video_name = self.splits[self.split_index][self.mode+'_keys'][index]
    frame_features = torch.Tensor(np.array(self.video_data[video_name + '/features']))
    if self.mode == 'test':
      return frame_features, video_name
    else:
      return frame_features

def get_loader(mode, split_index):
  if mode.lower() == 'train':
    vd = VideoData(mode, split_index)
    return DataLoader(vd, batch_size=1)
  else:
    return VideoData(mode, split_index)
```

##Train our model


```python
INPUT_SIZE = 1024
HIDDEN_SIZE = 512
LEARNING_RATE = 1e-4
DISCRIMINATOR_LEARNING_RATE = 1e-5
REGULARIZATION_FACTOR = 0.15
MODE = 'train'
N_EPOCHS = 10
CLIP = 0.1

original_label = torch.tensor(1.0).cuda()
summary_label = torch.tensor(0.0).cuda()


class Solver:

  def __init__(self, train_loader=None, test_loader=None):
    self.train_loader = train_loader
    self.test_loader = test_loader
  
  def build(self):
    """
    Set up the architecture's modules and define the losses used.
    """
    # MSELoss
    self.criterion = nn.MSELoss()

    # Build the baseline's modules
    self.linear_compress = nn.Linear(
        INPUT_SIZE,
        HIDDEN_SIZE
    ).cuda()

    # Instantiate the Summarizer
    self.summarizer = Summarizer(
        input_size=HIDDEN_SIZE,
        hidden_size=HIDDEN_SIZE,
    ).cuda()

    # Intantiate the Discriminator
    self.discriminator = Discriminator(
        input_size = HIDDEN_SIZE,
        hidden_size = HIDDEN_SIZE
    ).cuda()

    self.model = nn.ModuleList(
        [self.linear_compress, self.summarizer, self.discriminator]
    )

    # Define the optimizers
    if MODE == 'train':
      # Updates Linear Compress, eLSTM, sLSTM
      self.s_e_optimizer = optim.Adam(
          list(self.summarizer.s_lstm.parameters())
          + list(self.summarizer.auto_enc.e_lstm.parameters())
          + list(self.linear_compress.parameters()),
          lr=LEARNING_RATE
      )

      # dLSTM + LC
      self.d_optimizer = optim.Adam(
          list(self.summarizer.auto_enc.d_lstm.parameters())
          + list(self.linear_compress.parameters()),
          lr=LEARNING_RATE
      )

      # cLSTM + LC
      self.c_optimizer = optim.Adam(
          list(self.discriminator.parameters())
          + list(self.linear_compress.parameters()),
          lr=DISCRIMINATOR_LEARNING_RATE
      )

      # Define the cost functions
  def reconstruction_loss(self, h_origin, h_sum):
    return torch.norm(h_origin - h_sum, p=2)

  def sparsity_loss(self, scores):
    return torch.abs(torch.mean(scores) - REGULARIZATION_FACTOR)

  # Define the training loop
  def train(self):
    step = 0

    # ======================================================================
    split_keys = self.train_loader.dataset.splits[self.train_loader.dataset.split_index]['train_keys']
    final_max_length = {}
    all_shot_lengths = {}
    # ======================================================================

    for epoch_i in trange(N_EPOCHS, desc='Epoch', ncols=80):
      s_e_loss_history = []
      d_loss_history = []
      c_original_loss_history = []
      c_summary_loss_history = []

      for batch_i, image_features in enumerate(tqdm(
            self.train_loader, desc="Batch", ncols=80, leave=False
      )):

        self.model.train()

        # Get the image features
        # Image features size: [seq_len, 1024]
        image_features = image_features.view(-1, INPUT_SIZE)

        # Copy the image features to the GPU
        image_features_ = Variable(image_features).cuda()

        #----------------------Train sLSTM, eLSTM---------------------------
        print('\nTraining sLSTM and eLSTM...')

        # [seq_len, 1, hidden_size]
        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
        
        scores, generated_features = self.summarizer(original_features)
        print("original_features: {}".format(original_features.size()))
        h_origin, original_prob = self.discriminator(original_features)
        print("generated_features: {}".format(generated_features.size()))
        h_sum, sum_prob = self.discriminator(generated_features)

        _, (e, c) = self.summarizer.auto_enc.e_lstm(original_features)

        tqdm.write(f'Original probability: {original_prob.item():.3f}, Summary probability: {sum_prob.item():.3f}')

        reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)

        sparsity_loss = self.sparsity_loss(scores)

        tqdm.write(f'Reconstruction loss {reconstruction_loss.item():.3f}, sparsity loss: {sparsity_loss.item():.3f}')
        s_e_loss = reconstruction_loss + sparsity_loss 

        self.s_e_optimizer.zero_grad()
        s_e_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.s_e_optimizer.step()

        s_e_loss_history.append(s_e_loss.data)

        #---------------------------------------------------------------

        #---------------------------------------------------------------
        # Train the dLSTM (generator)
        tqdm.write('\nTraining dLSTM...')

        # [seq_len, 1, hidden_size]
        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

        scores, generated_features = self.summarizer(original_features)
        h_origin, original_prob = self.discriminator(original_features)
        h_sum, sum_prob = self.discriminator(generated_features)

        # Print the output probabilities
        tqdm.write(f'Original probability: {original_prob.item():.3f}, Summary Probability: {sum_prob.item():.3f}')

        reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
        g_loss = self.criterion(sum_prob, original_label)
        # dict_loss = self.dict_criterion(e[-1, :, :], h_par)

        tqdm.write(f'Reconstruction Loss: {reconstruction_loss.item():.3f} Generator Loss: {g_loss.item():.3f}')

        d_loss = reconstruction_loss + g_loss

        self.d_optimizer.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.d_optimizer.step()

        d_loss_history.append(d_loss.data)

        # ---------------------------------------------------------------

        # ---------------------------------------------------------------
        # Train the cLSTM
        tqdm.write('\nTraining cLSTM...')

        self.c_optimizer.zero_grad()

        # Train with original loss
        # [seq_len, 1, hidden_size]
        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
        h_origin, original_prob = self.discriminator(original_features)
        c_original_loss = self.criterion(original_prob, original_label)
        c_original_loss.backward()

        # Train with summary loss
        scores, generated_features = self.summarizer(original_features)
        h_sum, sum_prob = self.discriminator(generated_features.detach())
        c_summary_loss = self.criterion(sum_prob, summary_label)
        c_summary_loss.backward()

        tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
        tqdm.write(f'gen loss: {g_loss.item():.3f}')

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.c_optimizer.step()

        c_original_loss_history.append(c_original_loss.data)
        c_summary_loss_history.append(c_summary_loss.data)

        step += 1

  def evaluate(self, epoch_i):
    self.model.eval()

    out_dict = {}

    for video_tensor, video_name in tqdm(self.test_loader,
                                         desc='Evaluate',
                                         ncols=80,
                                         leave=False):
      video_tensor = video_tensor.view(-1, INPUT_SIZE)
      video_feature = Variable(video_tensor).cuda()

      video_feature = self.linear_compress(video_feature.detach()).unsqueeze(1)

      with torch.no_grad():
        scores = self.summarizer.s_lstm(video_feature).squeeze(1)
        scores = scores.cpu().numpy().tolist()

        out_dict[video_name] = scores
      
      score_save_path = VIDEO_NAME+str(epoch_i)+'.json'
      print(score_save_path)
      with open(score_save_path, 'w') as f:
        json.dump(out_dict, f)
      # score_save_path.chmod(0o777)



```


```python
train_loader = get_loader(mode='train', split_index=0)
test_loader = get_loader(mode='test', split_index=0)
```


```python
solver = Solver(train_loader, test_loader)
solver.build()
```


```python
solver.evaluate(-1)
```

    Evaluate:  40%|██████████████                     | 2/5 [00:00<00:00,  9.43it/s]

    summe-1.json
    summe-1.json
    

                                                                                    

    summe-1.json
    summe-1.json
    summe-1.json
    

    


```python
solver.train()
```

    Epoch:   0%|                                             | 0/10 [00:00<?, ?it/s]
    Batch:   0%|                                             | 0/20 [00:00<?, ?it/s][A

    
    Training sLSTM and eLSTM...
    torch.Size([300, 1, 512])
    torch.Size([300, 1, 512])
    original_features: torch.Size([300, 1, 512])
    generated_features: torch.Size([300, 1, 512])
    

    
    
    Epoch:   0%|                                             | 0/10 [00:00<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:00<?, ?it/s]

    Original probability: 0.505, Summary probability: 0.504
    Reconstruction loss 0.072, sparsity loss: 0.350
    

    
    
    Epoch:   0%|                                             | 0/10 [00:00<?, ?it/s]

    
    Training dLSTM...
    torch.Size([300, 1, 512])
    

    
    
    Epoch:   0%|                                             | 0/10 [00:01<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:01<?, ?it/s]

    torch.Size([300, 1, 512])
    Original probability: 0.505, Summary Probability: 0.504
    Reconstruction Loss: 0.067 Generator Loss: 0.246
    

    
    
    Epoch:   0%|                                             | 0/10 [00:01<?, ?it/s]

    
    Training cLSTM...
    torch.Size([300, 1, 512])
    

    
    
    Epoch:   0%|                                             | 0/10 [00:02<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:02<?, ?it/s]
    Batch:   5%|█▊                                   | 1/20 [00:02<00:41,  2.19s/it][A

    torch.Size([300, 1, 512])
    original_p: 0.505, summary_p: 0.504
    gen loss: 0.246
    
    Training sLSTM and eLSTM...
    torch.Size([649, 1, 512])
    

    
                                                                                    

    torch.Size([649, 1, 512])
    original_features: torch.Size([649, 1, 512])
    generated_features: torch.Size([649, 1, 512])
    

    
    Epoch:   0%|                                             | 0/10 [00:02<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:02<?, ?it/s]

    Original probability: 0.505, Summary probability: 0.504
    Reconstruction loss 0.063, sparsity loss: 0.341
    

    
    
    Epoch:   0%|                                             | 0/10 [00:03<?, ?it/s]

    
    Training dLSTM...
    torch.Size([649, 1, 512])
    

    
    
    Epoch:   0%|                                             | 0/10 [00:04<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:04<?, ?it/s]

    torch.Size([649, 1, 512])
    Original probability: 0.505, Summary Probability: 0.504
    Reconstruction Loss: 0.058 Generator Loss: 0.246
    

    
    
    Epoch:   0%|                                             | 0/10 [00:05<?, ?it/s]

    
    Training cLSTM...
    torch.Size([649, 1, 512])
    

    
    
    Epoch:   0%|                                             | 0/10 [00:06<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:06<?, ?it/s]

    torch.Size([649, 1, 512])
    original_p: 0.505, summary_p: 0.504
    gen loss: 0.246
    

    
    Batch:  10%|███▋                                 | 2/20 [00:06<00:59,  3.28s/it][A
    
    Epoch:   0%|                                             | 0/10 [00:06<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:06<?, ?it/s]

    
    Training sLSTM and eLSTM...
    torch.Size([108, 1, 512])
    torch.Size([108, 1, 512])
    original_features: torch.Size([108, 1, 512])
    generated_features: torch.Size([108, 1, 512])
    Original probability: 0.504, Summary probability: 0.504
    Reconstruction loss 0.054, sparsity loss: 0.334
    

    
    
    Epoch:   0%|                                             | 0/10 [00:06<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:06<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:06<?, ?it/s]

    
    Training dLSTM...
    torch.Size([108, 1, 512])
    torch.Size([108, 1, 512])
    Original probability: 0.504, Summary Probability: 0.504
    Reconstruction Loss: 0.051 Generator Loss: 0.246
    

    
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    Batch:  15%|█████▌                               | 3/20 [00:07<00:38,  2.25s/it][A

    
    Training cLSTM...
    torch.Size([108, 1, 512])
    torch.Size([108, 1, 512])
    original_p: 0.504, summary_p: 0.504
    gen loss: 0.246
    
    Training sLSTM and eLSTM...
    

    
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]

    torch.Size([64, 1, 512])
    torch.Size([64, 1, 512])
    original_features: torch.Size([64, 1, 512])
    generated_features: torch.Size([64, 1, 512])
    Original probability: 0.505, Summary probability: 0.504
    Reconstruction loss 0.046, sparsity loss: 0.326
    

    
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]

    
    Training dLSTM...
    torch.Size([64, 1, 512])
    torch.Size([64, 1, 512])
    Original probability: 0.505, Summary Probability: 0.504
    Reconstruction Loss: 0.043 Generator Loss: 0.246
    

    
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:07<?, ?it/s]
    Batch:  20%|███████▍                             | 4/20 [00:07<00:26,  1.65s/it][A

    
    Training cLSTM...
    torch.Size([64, 1, 512])
    torch.Size([64, 1, 512])
    original_p: 0.505, summary_p: 0.504
    gen loss: 0.246
    
    Training sLSTM and eLSTM...
    torch.Size([213, 1, 512])
    

    
    
    Epoch:   0%|                                             | 0/10 [00:08<?, ?it/s]
    
    Epoch:   0%|                                             | 0/10 [00:08<?, ?it/s]

    torch.Size([213, 1, 512])
    original_features: torch.Size([213, 1, 512])
    generated_features: torch.Size([213, 1, 512])
    Original probability: 0.505, Summary probability: 0.504
    Reconstruction loss 0.043, sparsity loss: 0.317
    

    
    Epoch:   0%|                                             | 0/10 [00:08<?, ?it/s]
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-205-ad90b845b08b> in <module>()
    ----> 1 solver.train()
    

    <ipython-input-201-0064132277ce> in train(self)
        133         s_e_loss.backward()
        134         # Gradient clipping
    --> 135         torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        136         self.s_e_optimizer.step()
        137 
    

    /usr/local/lib/python3.7/dist-packages/torch/nn/utils/clip_grad.py in clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)
         41     else:
         42         total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    ---> 43     if total_norm.isnan() or total_norm.isinf():
         44         if error_if_nonfinite:
         45             raise RuntimeError(
    

    KeyboardInterrupt: 



```python
import torch
a_tensor = torch.rand(300, 2, 1, 512)
```


```python
a_tensor.size()
```


```python
a_tensor = a_tensor.squeeze()
```


```python
a_tensor.size()
```


```python

```
