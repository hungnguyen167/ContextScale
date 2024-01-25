import numpy as np
import time
import datetime
import torch
from sklearn.decomposition import PCA, TruncatedSVD
import umap.umap_ as umap
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_tags, strip_punctuation, strip_numeric, strip_multiple_whitespaces


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def party_deu(name):
    if any([x in name for x in ["Linke", "Sozialismus"]]):
        return "PDS/Die Linke"
    elif name.find("Sozialdemokratische") != -1:
        return "SPD"
    elif name.find("Christlich") != -1:
        return "CDU/CSU"
    elif name.find("Freie") != -1:
        return "FDP"
    elif name.find("Grünen") != -1:
        return "Die Grünen"
    elif name.find("Alternative") != -1:
        return "AfD"
    else:
        return "NA"

def extract_code(code):
    if len(code) > 3:
        return(int(code[:3]))
    else:
        return int(code)
    
def group_texts(dataset, label_cols, text_col, group_factor=5):
    results = []
    for name, group in dataset.groupby(label_cols):
        ls_texts = []
        print(f'Length of {name} is: {len(group)}')
        labels = '_'.join([str(element) for element in name])
        texts = group[text_col].tolist()
        total_length = len(texts)
        for i in range(0,total_length,group_factor):
            text_to_append = ' '.join(texts[i:i+group_factor])
            ls_texts.append(text_to_append)
        result = {
            'text': ls_texts,
            'labels': labels
        }                  
        results.append(result)
    return(results)   


def lrn_code(code_long, code_short):
    if code_short in [
        407, ## protectionism - negative
        603, ## traditional morality - positive
        204, ## constitutionalism: negative
        507, ## education limitation
        410, ## economic growth: positive
        110, ## european integration: negative
        302, ## centralization: positive
        601, ## national way of life: positive
        608, ## multiculturalism: negative
        505, ## welfare state limitation
        104, ## military: positive
        102, ## foreign special relationships: negative
        109, ## internationalism: negative
        414, ## economic orthodoxy: positive
        402, ## incentives: positive
        401, ## free enterprise: positive 

    ] or code_long in [
        70302, ## agriculture and farmers: negative
        60501 ## law and order: positive
    ]:                                                                                        
        return('right')
    elif code_short in [
        406, ## protectionism: positive
        604, ## traditional morality: negative
        203, ## constitutionalism: positive
        506, ## education expansion
        416, ## antigrowth economy: positive
        501, ## environment protection
        108, ## european integration: positive
        301, ## decentralization: positive
        602, ## national way of life: negative
        607, ## multiculturalism: positive
        504, ## welfare state expansion
        105, ## military negative
        101, ## foreign special relationships: positive
        107, ## internationalism: positive
        409, ## keynesian demand management: positive
        403, ## market regulation: positive
        404, ## economic planning: positive
    ] or code_long in [
        70301, ## agriculture and farmers: positive
        60502, ## law and order: negative
    ]:                                                                            
        return('left')
    else:
        return('neutral/unknown')
    

def coalition_lr(code_long, code_short):
    if code_short in [406,201,202,604,705,706,506,416,501,108,
                301,602,607,503,504,105,403,404,409,412,413,
                101,103,106,107] or code_long in [70301,60502]:
        return('left')
    elif code_short in [407,603,507,410,110,302,601,608,505,104,401,
                  402,414,102,109] or code_long in [70302, 60501]:
        return('right')
    else:
        return('neutral')
    

def subcode_trans(code):
    if code == '000':
        return('0')
    else: 
        return(str(int(float(code)*100)))
    

def coalition_topic(code_short):
    if code_short in [406,407,703]:
        return('Agriculture')
    elif code_short in [201,202,603,604,605,705,706]:
        return('Civil Rights')
    elif code_short in [506,507]:
        return('Education')
    elif code_short in [416,501,410]:
        return('Environment')
    elif code_short in [108,110]:
        return('European integration')
    elif code_short in [301,302]:
        return('Decentralization')
    elif code_short in [601,602,607,608]:
        return('Immigration')
    elif code_short in [503,504,505]:
        return('Social Welfare')
    elif code_short in [104,105]:
        return('Defense')
    elif code_short in [401,402,403,404,409,412,413,414]:
        return('Economy')
    elif code_short in [101,102,103,106,107,109]:
        return('International politics')
    else:
        return('Other')

def cmp_scale(dataframe,text_var, group_vars, lr_kws: dict, sent_var):
    relscale = []
    absscale = []
    logscale = []
    name_ls = []
    for name, group in dataframe.groupby(group_vars):
        text = group[text_var].tolist()
        len_all = len(text)
        left = group[group[sent_var] == lr_kws['left']][sent_var].tolist()
        right = group[group[sent_var] == lr_kws['right']][sent_var].tolist()
        relscale.append((len(right) - len(left)) / (len(left) + len(right)))
        absscale.append((len(right) - len(left)) / len_all)
        logscale.append(np.log(len(right) + 0.5) - np.log(len(left) + 0.5))
        name_ls.append(name)
    return (absscale, relscale, logscale, name_ls)




def scale_umap(
    dataframe,
    model_transformer,
    text_var,
    guide_var,
    random_state,
    scale_by_group=None,
    n_neighbors=None,
    batch_size=512,
    guidance_weight=0.5,
    n_components=2,
    use_labels=True,
    auto_k=True,
    multi=False,
    **kwargs
):
    enc = OrdinalEncoder()
    frames = []
    ls_embeds = []
    if multi is False: 
        texts = dataframe[text_var].tolist()
        document_embed = model_transformer.encode(
                texts, batch_size=batch_size, device="cuda"
            )
        print(f"The shape of this document embeddings is: {document_embed.shape}")
        if auto_k is True:
                if len(dataframe) <= 10000:
                    k_recommended = 250
                elif 10000 <= len(dataframe) <= 50000:
                    k_recommended = 500
                elif len(dataframe) >= 50000:
                    k_recommended = 750
        else:
                k_recommended = n_neighbors
            
        if use_labels is True:
                y = enc.fit_transform(
                    np.reshape(dataframe.loc[:, guide_var].tolist(), newshape=(-1, 1))
                )
        else:
                y = None
        umap_scaler = umap.UMAP(
                n_components=n_components,
                n_neighbors=k_recommended,
                n_epochs=100,
                target_weight=guidance_weight,
                low_memory=True,
                verbose=False,
                random_state=random_state,
                **kwargs
            )
        embeddings = umap_scaler.fit_transform(document_embed, y=y)
        df_merged = dataframe.copy()
        for i in range(n_components):
            df_merged[''.join(['umap_d',str(i+1)])] = embeddings[:,i]
        df_merged['y'] = y.flatten()
        ls_embeds.append(document_embed)
    else:
        for name, group in dataframe.groupby(scale_by_group):
            print(f'Start scaling parties in {name}')
            if len(group) < 10:
                continue
            else: 
                texts = group[text_var].tolist()
                document_embed = model_transformer.encode(
                    texts, batch_size=batch_size, device="cuda"
                )
                print(f"The shape of this document embeddings is: {document_embed.shape}")
                if auto_k is True:
                    if len(group) <= 10000:
                        k_recommended = 250
                    elif 10000 <= len(group) <= 50000:
                        k_recommended = 500
                    elif len(group) >= 50000:
                        k_recommended = 750
                else:
                    k_recommended = n_neighbors

                if use_labels is True:
                    y = enc.fit_transform(
                        np.reshape(group.loc[:, guide_var].tolist(), newshape=(-1, 1))
                    )
                else:
                    y = None
                umap_scaler = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=k_recommended,
                    n_epochs=100,
                    target_weight=guidance_weight,
                    low_memory=True,
                    verbose=False,
                    random_state=random_state,
                    **kwargs
                )
                embeddings = umap_scaler.fit_transform(document_embed, y=y)
                for i in range(n_components):
                    group[''.join(['umap_d',str(i+1)])] = embeddings[:,i]
                group['y'] = y.flatten()
                ls_embeds.append({name: document_embed})
                frames.append(group)

            df_merged = pd.concat(frames)
    
    return df_merged, ls_embeds



def clean_text(text, stopwords):
    custom_filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric, strip_multiple_whitespaces]
    text = preprocess_string(text, custom_filters)
    tokens = [w for w in text if w not in stopwords]
    return ' '.join(tokens)

def d2v_reduct(model):
    keys = [k for k in model.dv.index_to_key]
    M = model.vector_size
    P = len(keys)
    embed_dict = {}
    for i in range( P ):
        embed_dict.update({keys[i]: model.dv[keys[i]]})   
    return embed_dict

def retrieve_vectors(dataset, model):
    res = []
    for index, row in dataset.iterrows():
        if index % 10000 == 0 and index != 0:
            print(index)
        text = row['text_cleaned']
        infer = model.infer_vector(text.split())
        res.append(infer)
    return np.array(res)
def encode_embeds(dataframe, text_var, model, batch_size,**kwargs):
    texts = dataframe[text_var].tolist()
    document_embed = model.encode(
        texts,
        batch_size=batch_size,
        device="cuda",
        convert_to_numpy=True,
        normalize_embeddings=True,
        **kwargs,
    )
    return document_embed

def format_time(elapsed):

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_ae(dataloader, model_ae, model_cls,ae_optimizer, cls_optimizer,device, ae_lossf, pred_lossf):
    print("")
    print("Training...")

    # Measure how long the training epoch takes.
    t0 = time.time()
    train_loss = 0
    # Put the model into training mode.
    size = len(dataloader.dataset)
    model_ae.train()
    model_cls.train()
    # For each batch of training data...optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for batch_num, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k,v in batch.items()}
        _, decoded, inputs = model_ae(batch['embeddings'])
        logits = model_cls(batch['embeddings'])
        loss_ae = ae_lossf(decoded, inputs)
        loss_pred = pred_lossf(logits, batch['labels'].long())
        loss = loss_ae + loss_pred
        train_loss += loss.item()
        # Backpropagation
        loss.backward()
        ae_optimizer.step()
        cls_optimizer.step()
        # Report
        if batch_num % 20 == 0 and batch_num != 0:
            elapsed = format_time(time.time() - t0)
            current = batch_num * len(batch['labels'])
            train_loss_current = train_loss / batch_num
            print(
                f"loss: {train_loss_current:>7f}  [{current:>5d}/{size:>5d}]. Took {elapsed}"
            )

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Training epoch took: {:}".format(training_time))






def lowe_topic(code):
    if code in [101,102,107,109]:                                                                               
        return('international relations')
    elif code in [108,110]:                                                                             
        return('eu')
    elif code in [301,302]:                                                                         
        return('decentralisation')
    elif code in [406,407]:                                                                            
        return('protectionism')
    elif code in [409,414]:                                                                            
        return('keynesian')
    elif code in [607,608]:                                                                              
        return('multiculturalism')
    elif code in [701,702]:                                                                            
        return('labour groups')
    elif code in [504,505]:                                                                              
        return('welfare')
    elif code in [506,507]:                                                                              
        return('education')
    elif code in [301,302]:                                                                             
        return('decentralisation')
    elif code in [401,402,403,412,413,415]:                                                                            
        return('freemarket')
    elif code in [416,501,410]:                                                                             
        return('environment')
    elif code in [104,105]:
        return('military')
    elif code in [203,204]:
        return('constitutionalism')
    elif code in [601,602,603,604]:
        return('traditionalism')
    else:
        return('irrelevant')
    
def lowe_lr(code):
    if code in [101,108,301,406,409,504,506,607,701,
                401,402,                                                                     ## free market
                416,501,                                                                    ## environment
                104,107,203,601,603]:                                       ## social - liberal                                                  
        return('positive')
    elif code in [102,110,302,407,414,505,507,608,702,
                403,412,413,415,                                                                ## free market
                410,                                                                            ## environment
                105,109,204,602,604]:                                             ## social - liberal                                 
        return('negative')
    else:
        return('unknown')

def train_loop(dataloader, model, optimizer, scheduler, loss_fn_topic, loss_fn_lr, loss_fn_reconstruct, device):
    print("")
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Put the model into training mode. 
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    # For each batch of training data...optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for batch_num, batch in enumerate(dataloader):
      batch = {k: v.to(device) for k,v in batch.items()}
      optimizer.zero_grad()
      y_topic = batch['topic'].long()
      y_lr = batch['lr'].long()
      logits_topic, logits_lr, logits_reconstruct, roberta_output = model(input_ids = batch['input_ids'], 
                                      attention_mask = batch['attention_mask'])
      loss_topic = loss_fn_topic(logits_topic, y_topic)
      loss_lr = loss_fn_lr(logits_lr, y_lr)
      loss_reconstruct = loss_fn_reconstruct(logits_reconstruct, roberta_output)
      loss = 0.3*loss_topic + 0.5*loss_lr + 0.2*loss_reconstruct
      # Backpropagation
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
      train_loss += loss.item()
      # Report
      if batch_num % 1000 == 0 and batch_num != 0:
        elapsed = format_time(time.time() - t0)
        current_loss = train_loss/batch_num
        current = batch_num * len(batch['input_ids'])
        print(f"loss: {current_loss:>7f}  [{current:>5d}/{size:>5d}]. Took {elapsed}")
    

  
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Training epoch took: {:}".format(training_time))
def eval_loop(dataloader, model, loss_fn, loss_fn_lr):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct_topic, correct_sent = 0.0, 0.0, 0.0
  model.eval()
  with torch.no_grad():  
    for batch in dataloader:    
        batch = {k: v.to(device) for k,v in batch.items()}
        y_topic = batch['topic'].long()
        y_lr = batch['lr'].long()
        logits_topic, logits_lr = model(input_ids = batch['input_ids'], 
                                        attention_mask = batch['attention_mask'],
                                        token_type_ids=batch['token_type_ids'])
        loss_topic = loss_fn(logits_topic, y_topic)
        loss_lr = loss_fn_lr(logits_lr, y_lr)
        loss = loss_topic + loss_lr
        test_loss += loss
        correct_topic += (logits_topic.argmax(1) == batch['topic']).type(torch.float).sum().item()
        correct_sent += (logits_lr.argmax(1) == batch['lr']).type(torch.float).sum().item()

  test_loss /= num_batches
  correct_topic /= size
  correct_sent /= size
  correct = (correct_topic+correct_sent)/2
  accuracy = correct*100
  print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return(accuracy)