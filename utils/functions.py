import numpy as np
import time
import datetime
import torch
from sklearn.decomposition import PCA, TruncatedSVD
import umap.umap_ as umap
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
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
        print(name)
        text = group['text'].tolist()
        len_all = len(text)
        left = group[group[sent_var] =='left'][sent_var].tolist()
        right = group[group[sent_var] =='right'][sent_var].tolist()
        relscale.append((len(right)-len(left))/(len(left)+len(right)))
        absscale.append((len(right)-len(left))/len_all)
        logscale.append(np.log(len(right)+0.5) - np.log(len(left)+0.5))
        name_ls.append(name)
    return(absscale, relscale, logscale, name_ls)
    
def encode_embeds(dataframe, text_var, model,**kwargs):
    texts = dataframe[text_var].tolist()
    document_embed = model.encode(texts,batch_size=512,device='cuda',convert_to_numpy=True,
                                  normalize_embeddings=True, **kwargs)
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
def scale_topic(dataframe, model_transformer, text_var, group_var, guide_var, 
                max_size=100000,guidance_weight=0.5, **kwargs):
    enc = OrdinalEncoder()
    for name, group in dataframe.groupby(group_var):
        pca = PCA(n_components=50)
        group = group.reset_index()
        col_idx = group.columns.get_loc(guide_var)
        print(f'Now scaling texts in: {name}')
        texts = group[text_var].tolist()
        document_embed = model_transformer.encode(texts, batch_size=512, normalize_embeddings=True, device='cuda')
        print(f'The shape of this document embeddings is: {document_embed.shape}')

    
    if len(dataframe) <= 10000:
        k_recommended = 250
    elif 10000 <= len(dataframe) <= 50000:
        k_recommended = 500
    elif len(dataframe) >=50000:
        k_recommended = 750


    if len(dataframe) >= max_size: ## tune max size according to your hardware
        idx = np.random.randint(document_embed.shape[0], size=max_size)
        sample_embed = document_embed[idx,:]
        y = enc.fit_transform(np.reshape(dataframe.iloc[idx,col_idx].tolist(), newshape=(-1,1)))
        embeddings_reduced_s = pca.fit_transform(sample_embed)
        embeddings_reduced_f = pca.fit_transform(document_embed)
        umap_fit = umap.UMAP(n_components=1, n_neighbors=k_recommended, n_epochs=250, metric='cosine', 
                            target_weight=guidance_weight, low_memory=True, verbose=True, **kwargs).fit(embeddings_reduced_s, y=y)
        embeddings = umap_fit.transform(embeddings_reduced_f)
        dataframe['policy_position'] = embeddings[:,0]
    else:
        y = enc.fit_transform(np.reshape(group.loc[:,guide_var].tolist(), newshape=(-1,1)))
        embeddings_reduced = pca.fit_transform(document_embed)
        embeddings = umap.UMAP(n_components=1, n_neighbors=k_recommended, n_epochs=250, metric='cosine', 
                            target_weight=guidance_weight, low_memory=True, verbose=True, **kwargs).fit_transform(embeddings_reduced, y=y)
        group['policy_position'] = embeddings[:,0]
        frames.append(group)
        
    results = pd.concat(frames)
    results = results.drop(columns=[text_var])
    return results
        

        
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
