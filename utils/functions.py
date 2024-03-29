import numpy as np
import time
import datetime
import torch
from sklearn.decomposition import PCA, TruncatedSVD
import umap.umap_ as umap
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_tags, strip_punctuation, strip_numeric, strip_multiple_whitespaces
from sklearn.metrics.pairwise import cosine_similarity


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def party_deu(party_code):
    if party_code in ['41111','41112','41113']:
        return "Alliance 90/Greens"
    elif party_code in ['41220','41221','41222','41223']:
        return "The Left"
    elif party_code == '41320':
        return "SPD"
    elif party_code == '41420':
        return "FDP"
    elif party_code == '41521':
        return "CDU/CSU"
    elif party_code == '41953':
        return "AfD"
    else:
        return "Other"
    

def extract_code(code):
    if len(code) > 3:
        return(int(code[:3]))
    else:
        return int(code)
    
def group_texts(dataset, label_cols, text_col, max_group_factor):
    results = []
    for name, group in dataset.groupby(label_cols):
        ls_texts = []
        labels = '_'.join([str(element) for element in name])
        texts = group[text_col].tolist()
        total_length = len(texts)
        if total_length <= max_group_factor:
            text_to_append = ' '.join(texts)
            ls_texts.append(text_to_append)
        else:
            for j in range(3,max_group_factor+1):
                if total_length % j == 0:
                    for i in range(0, total_length, j):
                        text_to_append = ' '.join(texts[i:i+j])
                        ls_texts.append(text_to_append)
                elif (total_length % j != 0) & (j==max_group_factor): 
                    for i in range(0, total_length, max_group_factor):
                        text_to_append = ' '.join(texts[i:i+max_group_factor])
                        ls_texts.append(text_to_append)
                else:
                    continue
        result = {
            'text': ls_texts,
            'labels': labels
        }                  
        results.append(result)
    return(results)   







def lrn_code(code_long, code_short):
    if code_short in [
        406, ## protectionism: positive
        204, ## constitutionalism: negative
        604, ## traditional morality: negative
        705, ## Minority Groups: Positive
        706, ## Noneconomic Demographic Groups: Positive
        506, ## education expansion
        416, ## antigrowth economy: positive
        501, ## environment protection
        108, ## european integration: positive
        301, ## decentralization: positive
        602, ## national way of life: negative
        607, ## multiculturalism: positive
        503, ## Equality: Positive
        504, ## welfare state expansion
        105, ## military: negative
        403, ## market regulation: positive
        404, ## economic planning: positive
        409, ## keynesian demand management: positive
        412, ## Controlled Economy: Positive
        413, ## Nationalization: Positive
        101, ## foreign special relationships: positive
        103, ## Anti-Imperialism: Positive
        106, ## Peace: Positive
        107, ## internationalism: positive
        701, ## labour groups: positive
    ] or code_long in [
        70310, ## agriculture and farmers: positive
        60520, ## law and order: negative 
        20120  ## human rights
        
    ]:
        return('left')
    elif code_short in [
        407, ## protectionism - negative
        203, ## constitutionalism: positive
        603, ## traditional morality - positive
        507, ## education limitation
        410, ## economic growth: positive
        110, ## european integration: negative
        302, ## centralization: positive
        305, ## political authority: positive
        601, ## national way of life: positive
        608, ## multiculturalism: negative
        505, ## welfare state limitation
        104, ## military: positive
        102, ## foreign special relationships: negative
        109, ## internationalism: negative
        414, ## economic orthodoxy: positive
        402, ## incentives: positive
        401, ## free enterprise: positive 
        702, ## labour groups: negative
    ] or code_long in [
        70320, ## agriculture and farmers: negative
        60510, ## law and order: positive 
        20110  ## freedom 
    ]:
        return('right')
    else:
        return('neutral')
    

def topic_code(code_short):
    if code_short in [406,407,703]:
        return('Agriculture')
    elif code_short in [201,203,204,603,604,605,705,706]:
        return('Civil Rights')
    elif code_short in [506,507]:
        return('Education')
    elif code_short in [416,501,410]:
        return('Environment')
    elif code_short in [108,110]:
        return('European integration')
    elif code_short in [301,302,305]:
        return('Decentralization')
    elif code_short in [601,602,607,608]:
        return('Immigration')
    elif code_short in [503,504,505,701,702]:
        return('Social Welfare')
    elif code_short in [104,105]:
        return('Defense')
    elif code_short in [401,402,403,404,409,412,413,414]:
        return('Economy')
    elif code_short in [101,102,103,106,107,109]:
        return('International politics')
    else:
        return('Other')
def subcode_trans(code):
    if code == '000':
        return('0')
    else: 
        return(str(int(float(code)*100)))
    


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

def train_loop(dataloader, model, optimizer, scheduler, loss_fn_topic, loss_fn_lr, loss_fn_reconstruct, device, topic_var, lr_var, train_topic=True):
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
      if train_topic is True:
        y_topic = batch[topic_var].long()
      y_lr = batch[lr_var].long()
      logits_topic, logits_lr, logits_reconstruct, roberta_output = model(input_ids = batch['input_ids'], 
                                      attention_mask = batch['attention_mask'])
      
      loss_lr = loss_fn_lr(logits_lr, y_lr)
      loss_reconstruct = loss_fn_reconstruct(logits_reconstruct, roberta_output)
      if train_topic is True:
        loss_topic = loss_fn_topic(logits_topic, y_topic)
        loss = 0.4*loss_topic + 0.4*loss_lr + 0.2*loss_reconstruct
      else:
        loss = 0.7*loss_lr + 0.3*loss_reconstruct
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


def eval_loop(dataloader, model, loss_fn_topic, loss_fn_lr, loss_fn_reconstruct, device, topic_var, lr_var, train_topic=True):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct_topic, correct_sent = 0.0, 0.0, 0.0
  model.eval()
  with torch.no_grad():  
    for batch in dataloader:    
        batch = {k: v.to(device) for k,v in batch.items()}
        if train_topic is True:
            y_topic = batch[topic_var].long()
        y_lr = batch[lr_var].long()
        logits_topic, logits_lr, logits_reconstruct, roberta_output = model(input_ids = batch['input_ids'], 
                                      attention_mask = batch['attention_mask'])
        loss_lr = loss_fn_lr(logits_lr, y_lr)
        loss_reconstruct = loss_fn_reconstruct(logits_reconstruct, roberta_output)
        if train_topic is True:
            loss_topic = loss_fn_topic(logits_topic, y_topic)
            loss = 0.4*loss_topic + 0.4*loss_lr + 0.2*loss_reconstruct
            correct_topic += (logits_topic.argmax(1) == batch[topic_var]).type(torch.float).sum().item()
        else:
            loss = 0.7*loss_lr + 0.3*loss_reconstruct
        test_loss += loss
        
        correct_sent += (logits_lr.argmax(1) == batch[lr_var]).type(torch.float).sum().item()

  test_loss /= num_batches
  correct_topic /= size
  correct_sent /= size
  correct = (correct_topic+correct_sent)/2
  print(f"Test Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  print(f"Test Error: \n Accuracy - LR: {(correct_sent*100):>0.1f}, Avg loss: {test_loss:>8f} \n")
  print(f"Test Error: \n Accuracy - Topic: {(correct_topic*100):>0.1f}, Avg loss: {test_loss:>8f} \n")

def test_loop(dataloader, model, device):
  model.eval()
  res_topic = []
  res_lr = []
  with torch.no_grad():  
    for batch in dataloader:    
        batch = {k: v.to(device) for k,v in batch.items()}
        logits_topic, logits_lr,_,_= model(input_ids = batch['input_ids'], 
                                        attention_mask = batch['attention_mask'])
        pred_topic = logits_topic.argmax(1)
        pred_lr = logits_lr.argmax(1)
        res_topic.append(pred_topic)
        res_lr.append(pred_lr)
  return res_topic, res_lr


def scale_func(dataloader, model, device):
    model.eval()
    res_topic = []
    res_lr_softmax = []
    print('Start predicting labels...')
    with torch.no_grad():  
        for batch in dataloader:    
            batch = {k: v.to(device) for k,v in batch.items()}
            logits_topic, logits_lr,_,_= model(input_ids = batch['input_ids'], 
                                            attention_mask = batch['attention_mask'])
            pred_topic = logits_topic.argmax(1)
            lr_softmax = logits_lr.softmax(1)

            res_topic.append(pred_topic)
            res_lr_softmax.append(lr_softmax)
    
    pred_topics = torch.cat(res_topic, dim=0).cpu().detach().numpy()
    lr_softmax = torch.cat(res_lr_softmax, dim=0).cpu().detach().numpy()
    anchor_prob = np.array([[1.0,0.0,0.0]])
    position_scores = 1-cosine_similarity(anchor_prob, lr_softmax)
    return pred_topics, position_scores


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

def tokenize_function(dataset, tokenizer, text_var, max_length):
    return tokenizer(dataset[text_var], truncation=True, max_length=max_length)