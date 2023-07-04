import numpy as np
import time
import datetime
import torch
from sklearn.decomposition import PCA, TruncatedSVD
import umap.umap_ as umap

def format_time(elapsed):

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
def party_deu(name):
    if any([x in name for x in ['Linke','Sozialismus']]):
        return('PDS/Die Linke')
    elif name.find('Sozialdemokratische') != -1:
        return('SPD')
    elif name.find('Christlich') != -1:
        return('CDU/CSU')
    elif name.find('Freie') != -1:
        return('FDP')
    elif name.find('Grünen') != -1:
        return('Die Grünen')
    elif name.find('Alternative') != -1:
        return('AfD')
    else:
        return('NA')
    
def cmp_scale(dataframe, group_vars, sent_var):
    relscale = []
    absscale = []
    logscale = []
    name_ls = []
    for name, group in dataframe.groupby(group_vars):
        text = group['text'].tolist()
        len_all = len(text)
        left = group[group[sent_var] =='left'][sent_var].tolist()
        right = group[group[sent_var] =='right'][sent_var].tolist()
        relscale.append((len(right)-len(left))/(len(left)+len(right)))
        absscale.append((len(right)-len(left))/len_all)
        logscale.append(np.log(len(right)+0.5) - np.log(len(left)+0.5))
        name_ls.append(name)
    return(absscale, relscale, logscale, name_ls)
    
def encode_embeds(dataframe, group_vars, model):
    embed_dict = {}
    for name, group in dataframe.groupby(group_vars):
        text = group['text'].tolist()
        embeds = model.encode(text)
        embed_dict.update({name: embeds})
    document_embed = np.vstack(list(embed_dict.values()))
    return(document_embed)

def get_prediction(model, texts, tokenizer, device):
  res = []
  with torch.no_grad():
    for idx, x in enumerate(texts):
      if idx % 1000 == 0:
        print(idx)
      inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(device)
      # perform inference to our model
      outputs = model(**inputs)
      # get output probabilities by doing softmax
      logits = outputs['logits']
      argmax = logits.argmax(1)
      res.append(argmax)
  return res

def scale_topic(document_embed, guide_labels, sparse=False, guidance_weight=0.8, n_components=1, **kwargs):
    y = guide_labels
    if len(document_embed) <= 10000:
        k_recommended = 250
    elif 10000 <= len(document_embed) <= 50000:
        k_recommended = 500
    elif len(document_embed) >=50000:
        k_recommended = 1000
    if sparse:
        svd = TruncatedSVD(n_components=50)
        embeddings_reduced = svd.fit_transform(document_embed)
    else:
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(document_embed)
    embeddings = umap.UMAP(n_components=n_components, n_neighbors=k_recommended, n_epochs=500, metric='euclidean', 
                            target_weight=guidance_weight, low_memory=True, verbose=True, **kwargs).fit_transform(embeddings_reduced, y=y)

    return embeddings
        
def train_ae(dataloader, model, optimizer, device, ae_lossf):
    print("")
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()
    train_loss = 0
    # Put the model into training mode. 
    size = len(dataloader.dataset)
    model.train()
    # For each batch of training data...optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for batch_num, batch in enumerate(dataloader):
      batch = batch.to(device)
      inputs, _, decoded = model(batch)
      loss = ae_lossf(decoded, inputs)
      train_loss += loss.item()
      optimizer.zero_grad()
      # Backpropagation
      loss.backward() 
      optimizer.step()
      # Report
      if batch_num % 100 == 0 and batch_num !=0:
        elapsed = format_time(time.time() - t0)
        current = batch_num * len(batch[0])
        train_loss_current = train_loss/batch_num
        print(f"loss: {train_loss_current:>7f}  [{current:>5d}/{size:>5d}]. Took {elapsed}")     
  
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    
    print("")
    print("  Training epoch took: {:}".format(training_time))
