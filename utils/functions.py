import numpy as np
import time
import datetime
import torch
from torch.nn.functional import one_hot
def recode_topic(value):
  if value in [1,15,18]:
    return(0)
  elif 1 <value <= 4:
    return(value-1)
  elif value in [5,13]:
    return(4)
  elif 5<value<=10:
    return(value-1)
  elif value == 12:
    return(10)
  elif value == 14:
    return(11)
  elif value in [16,17]:
    return(value-4)
  elif value in [19,20]:
    return(value-5)
  else:
    return(16)

def party_new(party_name):
    if party_name == 'CDU':
        return('CDU/CSU')
    elif party_name in ['Die Linke', 'PDS']:
        return('PDS/DIE LINKE')
    elif party_name == 'Grünen':
        return('GRUENEN')
    else:
        return(party_name)

def gov_last(party, year):
    if (year in [1953, 1957, 1965, 1983, 1987, 1990, 1994, 1998, 2013] and party in ['CDU/CSU', 'FDP']) or \
        (year == 1961 and party == 'CDU/CSU') or \
        (year in [1972, 1976, 1980] and party in ['SPD', 'FDP']) or \
        (year in [2002, 2005] and party in ['SPD', 'GRUENEN']) or \
        (year in [1969,2009, 2017] and party in ['CDU/CSU', 'SPD']):
        return(1)
    else:
        return(0)
        
def opp_last(party, year):
    if (year in [1953, 1957, 1965] and party == 'SPD') or \
        (year == 1961 and party in ['SPD', 'FDP']) or \
        (year == 1969 and party == 'FDP') or \
        (year in [1972, 1976, 1980] and party == 'CDU/CSU') or \
        (year in [1983, 1987, 1990] and party in ['SPD', 'GRUENEN']) or \
        (year in [1994, 1998] and party in ['SPD', 'GRUENEN', 'DIE LINKE']) or \
        (year in [2002, 2005] and party in ['CDU/CSU', 'FDP', 'DIE LINKE']) or \
        (year == 2009 and party in ['FDP', 'DIE LINKE', 'GRUENEN']) or \
        (year == 2013 and party in ['SPD', 'DIE LINKE', 'GRUENEN']) or \
        (year == 2017 and party in ['DIE LINKE', 'GRUENEN']):
        return(1)
    else:
        return(0)
 
def lab_to_top(values):
  for value in values:
    if value == 0:
      return("Economics")
    elif value == 1:
      return("Civil Rights")
    elif value == 2:
      return("Health")
    elif value == 3:
      return("Agriculture")
    elif value == 4:
      return("Labor and Social Welfare")
    elif value == 5:
      return("Education")
    elif value == 6:
      return("Environment")
    elif value == 7:
      return("Energy")
    elif value == 8:
      return("Immigration")
    elif value == 9:
      return("Transportation")
    elif value == 10:
      return("Law and Crime")
    elif value == 11:
      return("Housing")
    elif value == 12:
      return("Defense")
    elif value == 13:
      return("Technology")
    elif value == 14:
      return("International Affairs")
    elif value == 15:
      return("Government Operations")
    else:
      return("Other")

def augment_iter(dataset, list_cols, text_col, n_aug, augmenter=None):
  new_texts = np.array(dataset[text_col])
  trimmed_dataset = dataset[list_cols] ## make sure to include also labels
  final_vars = trimmed_dataset.to_numpy()
  for idx, text in enumerate(new_texts):
    if idx % 100 == 0:
      print(f'Augmenting the {idx}th sentence.')
    aug_texts = augmenter.augment(text, n_aug)
    rep_vars = np.repeat(np.array(trimmed_dataset.iloc[idx]), n_aug).reshape((n_aug,len(list_cols)))
    new_texts = np.append(new_texts, aug_texts)
    final_vars = np.vstack([final_vars, rep_vars])
  return(new_texts, final_vars)

def format_time(elapsed):

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_meta(dataloader, model, loss_fn, optimizer, scheduler, device):
    print("")
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()
    train_loss = 0
    # Put the model into training mode. 
    size = len(dataloader.dataset)
    model.train()
    # For each batch of training data...optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for batch, item in enumerate(dataloader):
      list_keys = [x for i,x in enumerate(list(item)) if i not in [0,1,2,len(list(item))-1, len(list(item))-2]]
      extras =  torch.cat(tuple(item[key] for key in list_keys), dim=1).to(device)
      output = model(input_ids = item['input_ids'].to(device), attention_mask = item['attention_mask'].to(device), token_type_ids = item['token_type_ids'].to(device), extras = extras, year = item['year'].to(device))
      y = item['labels'].long()
      loss = loss_fn(output, y)
      optimizer.zero_grad()
      # Backpropagation
      loss.backward()
      train_loss += loss.item()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
     
      # Report
      if batch % 100 == 0 and batch !=0:
        elapsed = format_time(time.time() - t0)
        current = batch * len(item['input_ids'])
        train_loss_current = train_loss/batch
        print(f"loss: {train_loss_current:>7f}  [{current:>5d}/{size:>5d}]. Took {elapsed}")     
    scheduler.step()   
  
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Training epoch took: {:}".format(training_time))

def test_meta(dataloader, model, loss_fn, device):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0
  model.eval()
  with torch.no_grad():  
    for item in dataloader:        
        list_keys = [x for i,x in enumerate(list(item)) if i not in [0,1,2,len(list(item))-1, len(list(item))-2]]
        extras =  torch.cat(tuple(item[key] for key in list_keys), dim=1).to(device)
        output = model(input_ids = item['input_ids'], attention_mask = item['attention_mask'], token_type_ids = item['token_type_ids'], extras = extras, year=item['year'])
        y = item['labels'].long()
        test_loss += loss_fn(output, y).item()
        correct += (output.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  accuracy = correct*100
  print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return(accuracy)




def train_normal(dataloader, model, optimizer, scheduler):
    print("")
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Put the model into training mode. 
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    # For each batch of training data...optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for batch, item in enumerate(dataloader):
      optimizer.zero_grad()
      y = item['labels'].long()
      outputs = model(input_ids = item['input_ids'], attention_mask = item['attention_mask'], token_type_ids = item['token_type_ids'], labels=y)
      loss = outputs[0]
      # Backpropagation
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      train_loss += loss.item()
      # Report
      if batch % 100 == 0 and batch != 0:
        elapsed = format_time(time.time() - t0)
        current_loss = train_loss/batch
        current = batch * len(item['input_ids'])
        print(f"loss: {current_loss:>7f}  [{current:>5d}/{size:>5d}]. Took {elapsed}")
    scheduler.step()

  
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Training epoch took: {:}".format(training_time))


  

def test_normal(dataloader, model):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0
  model.eval()
  with torch.no_grad():  
    for item in dataloader:    
        y = item['labels'].long()
        outputs = model(input_ids = item['input_ids'], attention_mask = item['attention_mask'], token_type_ids = item['token_type_ids'], labels=y)
        loss = outputs[0]
        logits = outputs[1]
        test_loss += loss
        correct += (logits.argmax(1) == item['labels']).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  accuracy = correct*100
  print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return(accuracy)

def train_loop(dataloader, model, optimizer, scheduler, device):
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
      batch = {k: v.to(device) for k, v in batch.items()}
      output = model(batch['input_ids'],batch['attention_mask'])
      y = batch['labels'].long()
      loss_fct = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).to(device)
      loss = loss_fct(output, y)
      optimizer.zero_grad()
      # Backpropagation
      loss.backward()
      train_loss += loss.item()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()   

      # Report
      if batch_num % 100 == 0 and batch_num !=0:
        elapsed = format_time(time.time() - t0)
        current = batch_num * len(batch['input_ids'])
        train_loss_current = train_loss/batch_num
        print(f"loss: {train_loss_current:>7f}  [{current:>5d}/{size:>5d}]. Took {elapsed}")     
  
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Training epoch took: {:}".format(training_time))
    
def eval_loop(dataloader, model, loss_fct, device):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0
  model.eval()
  with torch.no_grad():  
    for batch in dataloader:        
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch['input_ids'],batch['attention_mask'])
        y = batch['labels'].long()
        test_loss += loss_fct(output, y).item()
        correct += (output.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  accuracy = correct*100
  print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return(accuracy)