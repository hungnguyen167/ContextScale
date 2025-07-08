import numpy as np
import time
import datetime
import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_tags, strip_punctuation, strip_numeric, strip_multiple_whitespaces
from sklearn.metrics.pairwise import cosine_similarity
from torch.amp import GradScaler, autocast
import torch.nn as nn
from spacy.lang.ga.stop_words import STOP_WORDS
from nltk.corpus import stopwords



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
        labels = ';'.join([str(element) for element in name])
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

def rile_code(cmp_code):
    if cmp_code in [
        '104', 
        '201', 
        '203', 
        '305', 
        '401',
        '402',
        '407', 
        '414', 
        '505', 
        '601',
        '603',
        '605',
        '606'
    ]:
        return 'right'
    elif cmp_code in [      
        '103', 
        '105', 
        '106', 
        '107', 
        '403', 
        '404',
        '406',
        '412',
        '413',
        '504',
        '506',
        '701',
        '202'
    ]:
        return 'left'
    else:
        return 'neutral'
def recode_tw(topic, stance):
    if topic == 'women march':
        if stance == 0:
            return 'right'
        else:
            return 'left'
    else:
        if stance == 0:
            return 'left'
        else:
            return 'right'



def sentiment_code(cmp_code):
    ls_left =[
        '406', ## protectionism: positive
        '604', ## traditional morality: negative
        '705', ## Minority Groups: Positive
        '706', ## Noneconomic Demographic Groups: Positive
        '506', ## education expansion
        '416', ## antigrowth economy: positive
        '416.1', ## Anti-Growth Economy: Positive
        '501', ## environment protection
        '108', ## european integration: positive
        '602', ## national way of life: negative
        '602.1', ## National Way of Life General: Negative
        '602.2', ## National Way of Life General: Immigration: Positive
        '607', ## multiculturalism: positive
        '607.1', ##  multiculturalism general: positive
        '607.2',
        '607.3',
        '504', ## welfare state expansion
        '105', ## military: negative
        '403', ## market regulation: positive
        '404', ## economic planning: positive
        '409', ## Keynesian Demand Management
        '412', ## Controlled Economy: Positive
        '413', ## Nationalization: Positive
        '101', ## foreign special relationships: positive
        '107', ## internationalism: positive
        '701', ## labour groups: positive
        '503', ## Equality
        '605.2', ## law and order: negative ,
        '703', ## Agriculture and Farmers, positive
        '703.1', ## Agriculture and Farmers, positive
        '201.2', ## Human Rights
        '302', ## Decentralization
        '103', ## Anti-imperialism
    ]
    ls_right =  [
        '407', ## protectionism - negative
        '603', ## traditional morality - positive
        '507', ## education limitation
        '410', ## economic growth: positive
        '110', ## european integration: negative
        '601', ## national way of life: positive
        '601.1', ## National Way of Life General: Positive
        '601.2', ## National Way of Life General: Immigration: Negative
        '608', ## Multiculturalism: negative
        '608.1', ## Multiculturalism General: Negative
        '608.2', ## Multiculturalism: Immigrants Assimilation
        '608.3', ## Multiculturalism: Indigenous rights: Negative
        '505', ## welfare state limitation
        '104', ## military: positive
        '102', ## foreign special relationships: negative
        '109', ## internationalism: negative
        '414', ## economic orthodoxy: positive
        '402', ## incentives: positive
        '401', ## free enterprise: positive 
        '702', ## labour groups: negative
        '605.1', ## law and order: positive
        '605',  ## law and order: positive
        '703.2', ## Agriculture and Farmers, negative
        '201.1', ## Freedom
        '305', ## Political Authority
        '301' ## Centralization
    ]
    if cmp_code in ls_left:
        return('left')
    elif cmp_code in ls_right:
        return('right')
    else:
        return('neutral')
    
def topic_code(cmp_code):
    if cmp_code in ['406','407','703','703.1','703.2']:
        return 'Agriculture - Protectionism'
    elif cmp_code in ['201','201.1','201.2','705','706','605','605.1','605.2','603','604','606']:
        return 'Fabrics of Society'
    elif cmp_code in ['301','302','305','303','304']:
        return 'Political System'
    elif cmp_code in ['506','507']:
        return 'Education'
    elif cmp_code in ['416','410','501','416.1','416.2']:
        return 'Environment - Growth'
    elif cmp_code in ['108','110']:
        return 'European Integration'
    elif cmp_code in ['601','602','607','608','601.1','601.2','602.1','602.2','607.1','607.2','607.3','608.1','608.2','608.3']:
        return 'Immigration'
    elif cmp_code in ['503','504','505','701','702', '704']:
        return 'Labour and Social Welfare'
    elif cmp_code in ['104','105']:
        return 'Military'
    elif cmp_code in ['401','402','403','404','405','408','409','411','412','413','414']:
        return 'Economics'
    elif cmp_code in ['101','102','107','109','103','106']:
        return 'International Relations'
    else:
        return('Other')
    

def sentiment_code_coalition(cmp_short, cmp_long):
    if cmp_long in [
        '70311', ## Agriculture and Farmers, positive
        '70312',
        '70313',
        '70314', 
        '20102', ## Human Rights
        '60521', ## law and order: negative 
        '60522',
        '60523',
        '60524',
        '60525',
        '41601', ## Anti-Growth Economy: Positive
    ] or cmp_short in [
        '406', ## protectionism: positive
        '604', ## traditional morality: negative
        '705', ## Minority Groups: Positive
        '706', ## Noneconomic Demographic Groups: Positive
        '506', ## education expansion
        '501', ## environment protection
        '108', ## european integration: positive
        '602', ## national way of life: negative     
        '607', ## multiculturalism: positive
        '504', ## welfare state expansion
        '105', ## military: negative
        '403', ## market regulation: positive
        '404', ## economic planning: positive
        '409', ## Keynesian Demand Management
        '412', ## Controlled Economy: Positive
        '413', ## Nationalization: Positive
        '101', ## foreign special relationships: positive
        '107', ## internationalism: positive
        '701', ## labour groups: positive
        '503', ## Equality
        '703', ## Agriculture and Farmers, positive
        '302', ## Decentralization
        '103', ## Anti-imperialism,
        '607', ##  multiculturalism general: positive
        '602' ## National Way of Life General: Negative
    ]:
        return 'left'
    elif cmp_long in [
        '70321', ## Agriculture and Farmers, negative
        '70322',
        '70323',
        '70324',
        '20101', ## Freedom
        '60511', ## law and order: positive
        '60512',
        '60513',
        '60514',
        '60515',   
    ] or cmp_short in [
        '407', ## protectionism - negative
        '603', ## traditional morality - positive
        '507', ## education limitation
        '410', ## economic growth: positive
        '110', ## european integration: negative
        '601', ## national way of life: positive
        '608', ## Multiculturalism: negative
        '505', ## welfare state limitation
        '104', ## military: positive
        '102', ## foreign special relationships: negative
        '109', ## internationalism: negative
        '414', ## economic orthodoxy: positive
        '402', ## incentives: positive
        '401', ## free enterprise: positive 
        '702', ## labour groups: negative
        '605',  ## law and order: positive
        '305', ## Political Authority
        '301', ## Centralization,
        '608', ## Multiculturalism General: Negative
        '601' ## National Way of Life General: Positive
    ]:
        return 'right'
    else:
        return 'neutral'
    
def topic_code_coalition(cmp_short):
    if cmp_short in ['406','407','703']:
        return 'Agriculture - Protectionism'
    elif cmp_short in ['201','705','706','605','603','604','606']:
        return 'Fabrics of Society'
    elif cmp_short in ['301','302','305','303','304']:
        return 'Political System'
    elif cmp_short in ['506','507']:
        return 'Education'
    elif cmp_short in ['416','410','501']:
        return 'Environment - Growth'
    elif cmp_short in ['108','110']:
        return 'European Integration'
    elif cmp_short in ['601','602','607','608']:
        return 'Immigration'
    elif cmp_short in ['504','505','701','702', '503', '704']:
        return 'Labour and Social Welfare'
    elif cmp_short in ['104','105']:
        return 'Military'
    elif cmp_short in ['401','402','403','404','405','408','409','411','412','413','414']:
        return 'Economics'
    elif cmp_short in ['101','102','107','109','103','106']:
        return 'International Relations'
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
        if len(left) + len(right) == 0:
            print(name)
        relscale.append((len(right) - len(left)) / (len(left) + len(right) + 1e-5))
        absscale.append((len(right) - len(left)) / len_all)
        logscale.append(np.log(len(right) + 0.5) - np.log(len(left) + 0.5))
        name_ls.append(name)
    return (absscale, relscale, logscale, name_ls)





def clean_text(text, stopwords):
    custom_filters = [lambda x: x.lower(), 
                      strip_tags, 
                      strip_punctuation, 
                      strip_numeric, 
                      strip_multiple_whitespaces]
    text = preprocess_string(text, custom_filters)
    tokens = [w for w in text if w not in stopwords]
    return ' '.join(tokens)

def clean_text_loop(df, country_var):
    outputs = []
    irish_stopwords = STOP_WORDS
    icelandic_stopwords = icelandic_stopwords = [
            "að", "var", "og", "en", "í", "sem", "á", "með", "um", "við", 
            "fyrir", "hún", "hann", "það", "þeir", "þær", "eru", "ekki", 
            "hafa", "verið", "þá", "sína", "sér", "þess", "þeirra", "er"
        ] 
    for idx, text in enumerate(df['text']):
        if idx % 10000 ==0:
            print(f'Cleaning the {idx}th sentence')
        country = df.loc[idx,country_var].lower()
        if country in ['france','belgium']:
            outputs.append(clean_text(text, stopwords.words('french')))
        elif country in ['germany','austria', 'switzerland']:
            outputs.append(clean_text(text, stopwords.words('german')))
        elif country == 'italy':
            outputs.append(clean_text(text, stopwords.words('italian')))
        elif country == 'spain':
            outputs.append(clean_text(text, stopwords.words('spanish')))
        elif country == 'denmark':
            outputs.append(clean_text(text, stopwords.words('danish')))
        elif country == 'portugal':
            outputs.append(clean_text(text, stopwords.words('portuguese')))
        elif country == 'netherlands':
            outputs.append(clean_text(text, stopwords.words('dutch')))
        elif country == 'norway':
            outputs.append(clean_text(text, stopwords.words('norwegian')))
        elif country == 'sweden':
            outputs.append(clean_text(text, stopwords.words('swedish')))
        elif country == 'greece':
            outputs.append(clean_text(text, stopwords.words('greek')))
        elif country == 'ireland':
            outputs.append(clean_text(text, irish_stopwords)) 
        elif country == 'iceland':
            outputs.append(clean_text(text, icelandic_stopwords)) 
        else:
            outputs.append(clean_text(text, stopwords.words('english')))
    return outputs
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





def train_loop(dataloader, model, optimizer, scheduler, device, criterion_sent=None, criterion_topic=None, 
               sentiment_var=None, topic_var=None, timing_log=True):
    """
    Train the model for one epoch.

    Parameters:
        dataloader: DataLoader object for training data.
        model: PyTorch model to train.
        optimizer: Optimizer for model parameters.
        scheduler: Learning rate scheduler.
        device: Device (e.g., 'cuda' or 'cpu').
        criterion_sent: Loss function for sentiment classification.
        criterion_topic: Loss function for topic classification.
        sentiment_var: Key to access sentiment labels in the batch.
        topic_var: Key to access topic labels in the batch.
        timing_log: Whether to track timing.

    Returns:
        train_loss: Average loss over the training epoch.
        training_time: Total time taken for training.
    """
    print("")
    print('Training...')
    t0 = time.time()
    scaler = GradScaler()
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0.0
    batch_times = []

    for batch_num, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch_start = time.time()

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        topic_labels = batch.get(topic_var, None)
        sent_labels = batch.get(sentiment_var, None)

        # Forward pass and loss computation
        with autocast(device_type='cuda'):
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )

            # Sentiment loss
            if sentiment_var and sent_labels is not None and criterion_sent is not None:
                y_sent = sent_labels.long()
                loss_sent = criterion_sent(outputs['logits_sentiment'], y_sent)
            else:
                loss_sent = 0.0

            # Topic loss
            if topic_var and topic_labels is not None and criterion_topic is not None:
                y_topic = topic_labels.long()
                loss_topic = criterion_topic(outputs['logits_topic'], y_topic)
            else:
                loss_topic = 0.0

            # Combine losses
            loss = loss_sent + loss_topic
            train_loss += loss.item()

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()

        # Record batch time
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Print progress every 100 batches
        if batch_num % 100 == 0 and batch_num != 0:
            elapsed = time.time() - t0
            avg_batch_time = sum(batch_times) / len(batch_times)
            estimated_total_time = avg_batch_time * len(dataloader)
            estimated_remaining_time = estimated_total_time - elapsed
            current_loss = train_loss / (batch_num + 1)
            print(f"Batch {batch_num}: loss={current_loss:.6f}, elapsed={elapsed:.2f}s, remaining={estimated_remaining_time:.2f}s.")

    # Record total training time
    training_time = time.time() - t0
    print(f"\nTraining epoch took: {training_time:.2f}s")

    outputs = {}
    if timing_log is True:
        outputs['epoch_time'] = training_time
        outputs['avg_batch_time'] = sum(batch_times) / len(batch_times)

    return outputs




def eval_loop(dataloader, model, device, criterion_sent=None, criterion_topic=None, sentiment_var=None, topic_var=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct_topic, correct_sent = 0.0, 0.0, 0.0
    model.eval()
    
    with torch.no_grad():  
        for batch in dataloader:    
            batch = {k: v.to(device) for k, v in batch.items()}            
            topic_labels = batch.get(topic_var, None)
            sent_labels = batch.get(sentiment_var, None)

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
             # Compute topic classification loss and accuracy
            if topic_var is not None and topic_labels is not None and criterion_topic is not None:
                y_topic = topic_labels.long()  # Ground truth topic labels
                loss_topic = criterion_topic(outputs['logits_topic'], y_topic)
                correct_topic += (outputs['logits_topic'].argmax(1) == y_topic).type(torch.float).sum().item()
            else:
                loss_topic = 0.0

             # Compute sentiment loss and accuracy
            if sentiment_var is not None and sent_labels is not None and criterion_sent is not None:
                y_sent = sent_labels.long()                  
                loss_sent = criterion_sent(outputs['logits_sentiment'], y_sent)
                correct_sent += (outputs['logits_sentiment'].argmax(1) == y_sent).type(torch.float).sum().item()
            else:
                loss_sent = 0.0

           
            # Combine losses
            loss = loss_topic + loss_sent
            test_loss += loss

    # Calculate the average loss and accuracy
    test_loss /= num_batches
    correct_topic /= size
    correct_sent /= size
    correct = (correct_topic + correct_sent) / 2

    print(f"Test Error: \n Accuracy: {(correct * 100):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Accuracy - Sentiment: {(correct_sent * 100):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Accuracy - Topic: {(correct_topic * 100):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def test_loop(dataloader, model, device, topic_var=None, sentiment_var=None):
    model.eval()
    res_topic = []
    res_sentiment = []
    true_topics = []
    true_sentiments = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            topic_labels = batch.get(topic_var, None)
            sent_labels = batch.get(sentiment_var, None)

            # Forward pass through the model (ignore topic labels in test)
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )

            # Predict topics
            pred_topic = outputs['logits_topic'].argmax(1)  # Shape: [batch_size]
            res_topic.append(pred_topic)

            # Initialize a tensor to hold sentiment predictions, defaulting to -1
            pred_sentiment = outputs['logits_sentiment'].argmax(1)  # Shape: [batch_size]
            res_sentiment.append(pred_sentiment)

            if topic_labels is not None:
                true_topics.append(topic_labels)
            if sent_labels is not None:
                true_sentiments.append(sent_labels)

    # Concatenate results for topics and sentiments
    pred_topics = torch.cat(res_topic, dim=0).cpu().detach().numpy()
    pred_sentiments = torch.cat(res_sentiment, dim=0).cpu().detach().numpy()
    true_topics = torch.cat(true_topics, dim=0).cpu().detach().numpy() if len(true_topics) > 0 else None
    true_sentiments = torch.cat(true_sentiments, dim=0).cpu().detach().numpy() if len(true_sentiments) > 0 else None

    # Compute metrics for topics
    if true_topics is not None:
        precision_topic, recall_topic, f1_topic, _ = precision_recall_fscore_support(true_topics, pred_topics, average=None)
        matrix_topic = confusion_matrix(true_topics, pred_topics)
        accuracy_topic = matrix_topic.diagonal() / matrix_topic.sum(axis=1)

        # Prepare result table for topics
        res_table_topic = pd.DataFrame({
            'f1': np.round(f1_topic, 2),
            'precision': np.round(precision_topic, 2),
            'recall': np.round(recall_topic, 2),
            'accuracy': np.round(accuracy_topic, 2)
        })
    else:
        res_table_topic = None

    # Filter valid sentiment predictions (non -1)
    if true_sentiments is not None:
        precision_sentiment, recall_sentiment, f1_sentiment, _ = precision_recall_fscore_support(true_sentiments, pred_sentiments, average=None)
        matrix_sentiment = confusion_matrix(true_sentiments, pred_sentiments)
        accuracy_sentiment = matrix_sentiment.diagonal() / matrix_sentiment.sum(axis=1)

        # Prepare result table for sentiments
        res_table_sentiment = pd.DataFrame({
            'f1': np.round(f1_sentiment, 2),
            'precision': np.round(precision_sentiment, 2),
            'recall': np.round(recall_sentiment, 2),
            'accuracy': np.round(accuracy_sentiment, 2)
        })
    else:
        res_table_sentiment = None

    return res_table_topic, res_table_sentiment


def scale_func(dataloader, 
               model, 
               device, 
               topic_label=None, 
               sentiment_label=None, 
               timing_log=True,
               use_ground_truth_topic=False):
    model.eval()
    res_topic = []
    res_sentiment_sigmoid = []
    res_sentiment = []
    true_topics = []
    true_sentiments = []
    t0 = time.time()
    size = len(dataloader)
    print('Start predicting labels...')
    
    with torch.no_grad():
        for batch_num, batch in enumerate(dataloader):
            if topic_label in batch and batch_num == 0 and use_ground_truth_topic is True:
                print(f'Labels for topic are provided. They will be used for position scaling!')
            elif topic_label not in batch and batch_num == 0 and use_ground_truth_topic is False:
                print('Labels for topic are not provided. Using predicted topic labels for position scaling instead!')
            
            batch = {k: v.to(device) for k, v in batch.items()}
            topic_labels = batch.get(topic_label, None)
            sent_labels = batch.get(sentiment_label, None)

            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            # Predict topic labels
            pred_topic = outputs['logits_topic'].argmax(1)  
            res_topic.append(pred_topic)

            # Predict sentiment labels
            pred_sentiment = outputs['logits_sentiment'].argmax(1) 
            res_sentiment.append(pred_sentiment)

            # Apply sigmoid to logits_sentiment
            sentiment_sigmoid = torch.sigmoid(outputs['logits_sentiment'])  
            res_sentiment_sigmoid.append(sentiment_sigmoid)

            # Store true labels for evaluation
            if topic_labels is not None:
                true_topics.append(topic_labels)
            if sent_labels is not None:
                true_sentiments.append(sent_labels)

            if (batch_num + 1) % 1000 == 0:
                elapsed = time.time() - t0
                avg_batch_time = elapsed / (batch_num + 1)
                estimated_total_time = avg_batch_time * len(dataloader)
                estimated_remaining_time = estimated_total_time - elapsed
                print(f"Elapsed time: {elapsed:.2f}s, Estimated remaining time: {estimated_remaining_time:.2f}s")

    # Concatenate predictions across batches
    pred_topics = torch.cat(res_topic, dim=0).cpu().detach().numpy()
    pred_sentiment = torch.cat(res_sentiment, dim=0).cpu().detach().numpy()
    sentiment_sigmoid = torch.cat(res_sentiment_sigmoid, dim=0).cpu().detach().numpy()
    true_topics = torch.cat(true_topics, dim=0).cpu().detach().numpy() if len(true_topics) > 0 else None
    true_sentiments = torch.cat(true_sentiments, dim=0).cpu().detach().numpy() if len(true_sentiments) > 0 else None

    print('Start computing position scores')

    # Compute position scores
    if use_ground_truth_topic is True:
        topic_for_scaling = true_topics
    else:
        topic_for_scaling = pred_topics
    position_scores = np.zeros(len(topic_for_scaling))
    original_indices = np.arange(len(topic_for_scaling))
    min_adjustment = 0.3
    for topic_id in np.unique(topic_for_scaling):
        topic_mask = topic_for_scaling == topic_id
        topic_indices = original_indices[topic_mask]
        if sentiment_sigmoid.shape[1] == 3:
            left_values = sentiment_sigmoid[topic_mask][:, 0]
            neutral_values = sentiment_sigmoid[topic_mask][:, 1]
            right_values = sentiment_sigmoid[topic_mask][:, 2]

            neutral_range = np.max(neutral_values) - np.min(neutral_values)
            if neutral_range > 0:
                raw_adjustment = 1 - (neutral_values - np.min(neutral_values)) / neutral_range

            else:
                raw_adjustment = 1 - neutral_values  
            adjustment = min_adjustment + (1 - min_adjustment) * raw_adjustment

            position_scores[topic_indices] = (right_values - left_values) * adjustment
            
        elif sentiment_sigmoid.shape[1] == 2:
            left_values = sentiment_sigmoid[topic_mask][:, 0]
            right_values = sentiment_sigmoid[topic_mask][:, 1]
            position_scores[topic_indices] = right_values - left_values
        else:
            print('Function supports only 2 or 3 categories. Aborting!')
            position_scores = None

    # Compute metrics for topics
    if true_topics is not None:
        precision_topic, recall_topic, f1_topic, _ = precision_recall_fscore_support(true_topics, pred_topics, average=None)
        matrix_topic = confusion_matrix(true_topics, pred_topics)
        accuracy_topic = matrix_topic.diagonal() / matrix_topic.sum(axis=1)

        # Prepare result table for topics
        res_table_topic = pd.DataFrame({
            'f1': np.round(f1_topic, 2),
            'precision': np.round(precision_topic, 2),
            'recall': np.round(recall_topic, 2),
            'accuracy': np.round(accuracy_topic, 2)
        })
    else:
        res_table_topic = None

    # Compute metrics for sentiments
    if true_sentiments is not None:
        precision_sentiment, recall_sentiment, f1_sentiment, _ = precision_recall_fscore_support(true_sentiments, pred_sentiment, average=None)
        matrix_sentiment = confusion_matrix(true_sentiments, pred_sentiment)
        accuracy_sentiment = matrix_sentiment.diagonal() / matrix_sentiment.sum(axis=1)

        # Prepare result table for sentiments
        res_table_sentiment = pd.DataFrame({
            'f1': np.round(f1_sentiment, 2),
            'precision': np.round(precision_sentiment, 2),
            'recall': np.round(recall_sentiment, 2),
            'accuracy': np.round(accuracy_sentiment, 2)
        })
    else:
        res_table_sentiment = None

    # Record timing
    total_time = time.time() - t0

    outputs = {
        'res_table_topic': res_table_topic,
        'res_table_sentiment': res_table_sentiment,
        'position_scores': position_scores,
        'pred_topics': pred_topics,
        'pred_sentiment': pred_sentiment
    }
    if timing_log is True:
        outputs['total_time'] = total_time
        outputs['avg_batch_time'] = total_time / size

    return outputs







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





def tokenize_function(dataset, tokenizer, text_var, max_length):
    return tokenizer(dataset[text_var], truncation=True, padding=False, max_length=max_length)


def check_weights_similar(source_model, target_model, patterns):
    all_parameters_copied = True  # Flag to track if all parameters are copied successfully

    for name, source_param in source_model.named_parameters():
        if not name.startswith(patterns):  # Skip the layers you don't want to copy
            # Extract the corresponding parameter from the target model
            target_param = target_model
            for attr in name.split('.'):
                target_param = getattr(target_param, attr)
            
            # Check if the source and target parameters are identical
            if not torch.equal(source_param.data, target_param.data):
                print(f"Parameter {name} not copied correctly.")
                all_parameters_copied = False
                break  # Optional: stop checking after the first mismatch

    if all_parameters_copied:
        print("All parameters copied successfully.")
    else:
        print("Some parameters were not copied successfully.")


def copy_weights(source_model, target_model, patterns, freeze_copied=True):
    # Get the state dictionaries of both models
    initial_state_dict = source_model.state_dict()
    scaled_state_dict = target_model.state_dict()
    
    for name, param in initial_state_dict.items():
        if name in scaled_state_dict and not name.startswith(patterns):
            # Ensure dimensions match before copying
            if param.size() == scaled_state_dict[name].size():
                # Copy the parameter from source to target
                scaled_state_dict[name].copy_(param)
                
                # Freeze the layer if freeze_copied is True
                if freeze_copied:
                    # Access the parameter in the model's named_parameters() and set requires_grad
                    for target_name, target_param in target_model.named_parameters():
                        if target_name == name:
                            target_param.requires_grad = False
                            break
            else:
                print(f"Dimension mismatch for layer {name}, skipping.")
        else:
            print(f"Skipping {name} as it is not present or should be skipped in the scaling model.")

    # Optionally print trainable parameters to confirm
    print("Trainable Parameters after copying:")
    for name, param in target_model.named_parameters():
        if param.requires_grad:
            print(name)



def get_architecture_details(model):
    architecture = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and name != "":
            module_details = {
                'name': name,
                'type': type(module).__name__,
                'params': sum(p.numel() for p in module.parameters())
            }
            architecture.append(module_details)
    return architecture

def compare_architectures(arch1, arch2):
    if len(arch1) != len(arch2):
        print("The models have different number of layers/modules.")
    else:
        print("The models have the same number of layers/modules.")

    for layer1, layer2 in zip(arch1, arch2):
        if layer1 != layer2:
            print(f"Difference found in layer {layer1['name']}:")
            print(f"Model 1: {layer1}")
            print(f"Model 2: {layer2}")
       
    

