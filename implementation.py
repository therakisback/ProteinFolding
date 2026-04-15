
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm
import os.path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import linecache
import ast
import time
from transformers import T5EncoderModel, T5Tokenizer
from datetime import datetime

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available, using CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAXLENGTH = 200

def applyThreshold(tensor, threshold):  
    map = []
    for i, line in enumerate(tensor, 0):
        temp_line = []
        for element in line:
            if (element < threshold):
                temp_line.append(1.)
            else:
                temp_line.append(0.)
        map.append(temp_line)
    return map

def printMap(map, size = -1):
    if size > 0:
        newMap = map[:size][:size]
    else:
        newMap = map
    fig, ax = plt.subplots()
    ax.imshow(np.matrix(newMap), cmap='gray')
    ax.axis('off')
    plt.show()

def generate_data():
    data = {"name": [], "length": [], "proteins": [], "coord": []}
    print("Reading in from dataset files")

    # Read in data from the two files
    with open("pdb.fasta_qual.16Nov2022_30.fasta") as pdb_fasta:
        fasta_lines = pdb_fasta.readlines()[0::2]
    fasta_lines = [s.replace('>', '') for s in fasta_lines]
    fasta_lines = [s.replace('\n', '') for s in fasta_lines]
    
    i = 0

    # Convert data to a usable format and put it in a dictionary
    with open("pdb.fasta_qual.16Nov2022.adataset") as pdb_adataset:
        while True:
            segment = list(islice(pdb_adataset, 16))
            if not segment: # If there are no more lines we break the loop.
                break

            name = segment[0].replace("\n", "")
            if name in fasta_lines:
                length = int(segment[1])
                if (length <= MAXLENGTH):
                    data["name"].append(name)
                    data["length"].append(length)

                    # Residue data
                    proteins = segment[2].replace("\n", "").split("\t")
                    del proteins[-1]
                    data["proteins"].append(proteins)

                    # Coordinate data
                    coords = segment[13].replace("\n", "").split("\t")
                    del coords[-1]

                    i = 0
                    for string in coords:
                        coords[i] = list(map(float, string.split(" ")))
                        i += 1
                    data["coord"].append(coords) 
    print("Finished reading from files.")   

    print(f"Name: {data['name'][0]} \tLength: {data['length'][0]} \nResidue Chain: {data['proteins'][0]} \nCoordinates: {data['coord'][0]}")
    return data

class vhseDataset(Dataset):
    vhse_map = {
        'A': [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
        'R': [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.3, 0.83],
        'N': [-0.99, 0, -0.37, 0.69, -0.55, 0.85, 0.74, -0.8],
        'D': [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
        'C': [0.18, -1.67, -0.46, -0.21, 0, 1.2, -1.61, -0.19],
        'Q': [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.2, -0.41],
        'E': [-1.18, 0.4, 0.1, 0.36, -2.16, -0.17, 0.91, 0.02],
        'G': [-0.2, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
        'H': [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
        'I': [1.27, -0.14, 0.3, -1.8, 0.3, -1.61, -0.16, -0.13],
        'L': [1.36, 0.07, 0.26, -0.8, 0.22, -1.37, 0.08, -0.62],
        'K': [-1.17, 0.7, 0.7, 0.8, 1.64, 0.67, 1.63, 0.13],
        'M': [1.01, -0.53, 0.43, 0, 0.23, 0.1, -0.86, -0.68],
        'F': [1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -.02],
        'P': [0.22, -0.17, -0.5, 0.05, -0.01, -1.34, -0.19, 3.56],
        'S': [-0.67, -0.88, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
        'T': [-0.34, -0.51, -0.55, -1.06, 0.01, -0.01, -0.79, 0.39],
        'W': [1.5, 2.06, 1.79, 0.75, 0.75, -0.13, -1.06, -0.85],
        'Y': [0.61, 1.6, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52],
        'V': [0.76, -0.92, 0.17, -1.91, 0.22, -1.4, -0.24, -0.03],
        'X': [0, 0, 0, 0, 0, 0, 0, 0],
    }

    def __init__(self, num_data = -1, test = False):
        self.length = 0
        self.max = -1
        # If we are using all the data there is no need to seperate the two
        if num_data == -1:
            test = False
            
        if test:
            self.file = f"pdb-vhsetest{num_data}"
        else:
            self.file = f"pdb-vhse{num_data}"
        # Check to see if the necessary file exists, if it doesn't, create it
        if not os.path.isfile(self.file):
            # If the data hasn't been loaded from the file, do so.
            data = generate_data()

            # Cut data based on specified length. 'test' dataset takes from the end of the data
            if num_data >= 1:
                if test:
                    dataset = {key: value[-num_data:] for key, value in data.items()}
                else:
                    dataset = {key: value[:num_data] for key, value in data.items()}
            else:
                dataset = data
                
            
            with open(self.file, 'x') as pdb:
                for i, protein in enumerate(tqdm(dataset['proteins'])):
                    # Encode the protein residues based on the VHSE encodings dictionary
                    encoding = []
                    for residue in protein:
                        encoding.append(self.vhse_map[residue])

                    encoding = list(encoding)
                    pdb.write(f"{encoding}\n")

                    length = dataset['length'][i]
                    
                    # Convert residue coords to distances
                    protein = dataset['coord'][i]
                    protein = np.array(protein)
                    tensor = protein[:, None, :] - protein[None, :, :]
                    # Takes euclidean norm of  x,y,z
                    tensor = np.linalg.norm(tensor, axis=2)
                    tensor = applyThreshold(tensor, 8)
                    pdb.write(f"{tensor}\n")

                    pdb.write(f"{length}\n")

                    self.length += 1

        else:
            with open(self.file) as pdb:
                self.length = int(sum(1 for line in pdb) / 3)
    
    def __len__(self):
        return self.length
    
    # I have made this a drive-read class for memory reasons, it will be much slower but hopefully wont consume over 32gb of memory
    def __getitem__(self, index):
        f_i = (index * 3) + 1

        # Import encodings from file
        line = linecache.getline(self.file, f_i)
        protein = line.replace("\n", "")
        protein = np.array(ast.literal_eval(protein))
        protein = np.float32(protein)
        # I make a 32-length vector based on the pair of encodings
        tensor = np.concatenate([
                    protein[:, None, :] + np.zeros_like(protein[None, :, :]),  # features of first residue
                    protein[None, :, :] + np.zeros_like(protein[:, None, :]),  # features of second residue
                    protein[:, None, :] - protein[None, :, :],                 # difference
                    protein[:, None, :] * protein[None, :, :]                  # sum
                    ], axis=-1)
        tensor = torch.tensor(tensor)
        tensor = tensor.permute(2, 0, 1)
        tensor = torch.nn.functional.normalize(tensor)
        

        # Import length
        line = linecache.getline(self.file, f_i + 2)
        length = line.replace("\n", "")
        length = np.int64(length)

        # Import coordinate map
        line = linecache.getline(self.file, f_i + 1)
        map = line.replace("\n", "")
        map = np.array(ast.literal_eval(map))
        map = np.float32(map)
        map = torch.tensor(map).unsqueeze(0)

        linecache.checkcache()

        return tensor, map, length



def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer

#@title Generate embeddings. { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein,
                   max_residues=4000, max_seq_len=1000, max_batch=100 ):

    results = {"residue_embs" : dict(),
               "protein_embs" : dict(),
               "sec_structs" : dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            #token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            token_encoding = tokenizer(seqs, padding="longest", add_special_tokens=True, return_tensors="pt")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results

class ptDataset(Dataset):

    def __init__(self, num_data = -1, test = False):
        self.length = 0
        # If we are using all the data there is no need to seperate the two
        if num_data == -1:
            test = False
        if test:
            self.file = f"pdb-pttest{num_data}"
        else:
            self.file = f"pdb-pt{num_data}"
        
        if not os.path.isfile(self.file):

            data = generate_data()

            if num_data >= 1:
                if test:
                    dataset = {key: value[-num_data:] for key, value in data.items()}
                else:
                    dataset = {key: value[:num_data] for key, value in data.items()}
            else:
                dataset = data

            # I need to sort the dataset items in the same way as the embedding method so the data aligns
            indices = sorted(range(len(dataset['length'])), key=lambda i: dataset['length'][i], reverse=True)
            dataset = {key: [value[i] for i in indices] for key, value in dataset.items()}
                        
            # We need to convert the data to a dictionary that the prot-trans method can read
            # This should maintain order, so I can still use dataset
            sequences = {key: value for key, value in zip(dataset['name'], dataset['proteins'])}
            
            with open(self.file, 'x') as pdb:
                model, tokenizer = get_T5_model()
                # I am not sure how to implement the 1024x1 protein-embedding vector, so we will avoid it for now
                results = get_embeddings(model, tokenizer, sequences, True, False)
                for i, data in enumerate(tqdm(results['residue_embs'].items())):
                    key, value = data

                    # I write length first as it was important to retrieve first before, now the CNN is fully convolutional it's simply a vestigial remnant.
                    length = dataset['length'][i]
                    pdb.write(f"{length}\n")

                    pdb.write(f"{np.ndarray.tolist(value)}\n")

                    # Convert residue coords to distances
                    protein = dataset['coord'][i]
                    protein = np.array(protein)
                    tensor = protein[:, None, :] - protein[None, :, :]
                    tensor = np.linalg.norm(tensor, axis=2)
                    tensor = applyThreshold(tensor, 8)
                    pdb.write(f"{tensor}\n")
                    
                    self.length += 1    
        else:
            with open(self.file) as pdb:
                self.length = int(sum(1 for line in pdb) / 3)
                
                

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        f_i = (index * 3) + 1

        # Import length
        line = linecache.getline(self.file, f_i)
        length = line.replace("\n", "")
        length = np.int64(length)

        # Import and adjust encodings
        line = linecache.getline(self.file, f_i + 1)
        protein = line.replace("\n", "")
        protein = np.array(ast.literal_eval(protein))
        protein = np.float32(protein)
        tensor = np.concatenate([
                    protein[:, None, :] + np.zeros_like(protein[None, :, :]),  # first embedding
                    protein[None, :, :] + np.zeros_like(protein[:, None, :]),  # second embedding
                    protein[:, None, :] - protein[None, :, :],                 # difference
                    protein[:, None, :] * protein[None, :, :]                  # sum
                    ], axis=-1)
        tensor = torch.tensor(tensor)
        tensor = tensor.permute(2, 0, 1)
        tensor = torch.nn.functional.normalize(tensor)

        # Import coordinate map
        line = linecache.getline(self.file, f_i + 2)
        map = line.replace("\n", "")
        map = np.array(ast.literal_eval(map))
        map = np.float32(map)
        map = torch.tensor(map).unsqueeze(0)

        linecache.checkcache()

        return tensor, map, length

# Here we initialize both datasets, and their testing counterparts

train_data = vhseDataset(num_data = -1)
test_data = vhseDataset(num_data = 1000, test = True)

train_data_pt = ptDataset(num_data = -1)
test_data_pt = ptDataset(num_data = 1000, test = True)

# Generating PT embeddings can leave the GPU cache full, or at least messy.
torch.cuda.empty_cache()

class CNN(nn.Module):
    def __init__(self, feat, channels):
        super(CNN, self).__init__()

        self.prime = nn.Sequential(
            nn.Conv2d(feat, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.prime(x)
        x = self.hidden(x)
        x = self.final(x)
        
        x = (x + x.transpose(-1, -2)) / 2
        return x

loss_func = nn.BCEWithLogitsLoss().to(device)

def one_epoch(model, optimizer, train_loader):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        inputs, targets, length = data

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        
        loss = loss_func(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss # loss per batch
            print(f'  batch {i+1} loss: {last_loss}')
            running_loss = 0.
        
    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def save_checkpoint(state, filename):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def train_CNN(model, epochs, train_loader, test_loader, checkpoint_dir = "checkpoint"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_number = 0
    checkpoint_dir = checkpoint_dir + ".pth.tar"

    if os.path.isfile(checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir)
        load_checkpoint(checkpoint, model, optimizer)


    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)

        avg_loss = one_epoch(model, optimizer, train_loader)
        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(test_loader, 1):
                vinputs, vlabels, vlength = vdata
                vinputs = vinputs.to(device)
                voutputs = model(vinputs)
                vlabels = vlabels.to(device)
                vloss = loss_func(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f"Epoch {epoch}: LOSS train {avg_loss} valid {avg_vloss}")
        
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, checkpoint_dir)

        epoch_number += 1


# Train VHSE CNN
batch = 1
epochs = 5
train_loader = DataLoader(dataset = train_data, batch_size = batch, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = batch, shuffle = True)
vhse_model = CNN(32, 32)
vhse_model = vhse_model.to(device)

train_CNN(vhse_model, epochs, train_loader, test_loader, "vhse_chkpt")

# Train PT CNN

train_loader = DataLoader(dataset = train_data_pt, batch_size = batch, shuffle = True)
test_loader = DataLoader(dataset = test_data_pt, batch_size = batch, shuffle = True)
pt_model = CNN(4096, 64)
pt_model = pt_model.to(device)
# Epochs is defined above with the VHSE data. I will keep them the same for now.
train_CNN(pt_model, epochs, train_loader, test_loader, "pt_chkpt")

# To investigate performance of the CNN's check the test ipynb