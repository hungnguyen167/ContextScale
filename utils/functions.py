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
    if len(code.split('.')) > 1:
        return(int(code.split('.')[0]))
    else:
        return int(code)
    
def group_texts(dataset, labels_col, text_col, group_factor=5):
    results = []
    labels = dataset[labels_col].tolist()
    for label in set(labels):
        ls_texts = []
        current_label = dataset[dataset[labels_col] == label]
        print(f'Length of {label} is: {len(current_label)}')
        texts = current_label[text_col].tolist()
        total_length = len(texts)
        for i in range(0,total_length,group_factor):
            text_to_append = ' '.join(texts[i:i+group_factor])
            ls_texts.append(text_to_append)
        result = {
            'text': ls_texts,
            'labels': label
        }                  
        results.append(result)
    return(results)   


def lr_code(code):
    if code in [104,201,203,305,401,402,407,414,505,601,603,605,606,410]:                                                                                        
        return('right')
    elif code in [103,105,106,107,202,403,404,406,412,413,504,506,701,501,416]:                                                                            
        return('left')
    else:
        return('neutral')


def cmp_scale(dataframe, group_vars, sent_var):
    relscale = []
    absscale = []
    logscale = []
    name_ls = []
    for name, group in dataframe.groupby(group_vars):
        print(name)
        text = group["text"].tolist()
        len_all = len(text)
        left = group[group[sent_var] == "left"][sent_var].tolist()
        right = group[group[sent_var] == "right"][sent_var].tolist()
        relscale.append((len(right) - len(left)) / (len(left) + len(right)))
        absscale.append((len(right) - len(left)) / len_all)
        logscale.append(np.log(len(right) + 0.5) - np.log(len(left) + 0.5))
        name_ls.append(name)
    return (absscale, relscale, logscale, name_ls)

def cmp_scale_sentiment(dataframe, group_vars, sent_var):
    relscale = []
    absscale = []
    logscale = []
    name_ls = []
    for name, group in dataframe.groupby(group_vars):
        print(name)
        text = group["text"].tolist()
        neg = group[group[sent_var] == "negative"][sent_var].tolist()
        pos = group[group[sent_var] == "positive"][sent_var].tolist()
        len_all = len(text)
        relscale.append((len(pos) - len(neg)) / (len(neg) + len(pos)))
        absscale.append((len(pos) - len(neg)) / len_all)
        logscale.append(np.log(len(pos) + 0.5) - np.log(len(neg) + 0.5))
        name_ls.append(name)
    return (absscale, relscale, logscale, name_ls)

def cmp_scale(dataframe, group_vars, sent_var):
    relscale = []
    absscale = []
    logscale = []
    name_ls = []
    for name, group in dataframe.groupby(group_vars):
        print(name)
        text = group["text"].tolist()
        len_all = len(text)
        left = group[group[sent_var] == "left"][sent_var].tolist()
        right = group[group[sent_var] == "right"][sent_var].tolist()
        relscale.append((len(right) - len(left)) / (len(left) + len(right)))
        absscale.append((len(right) - len(left)) / len_all)
        logscale.append(np.log(len(right) + 0.5) - np.log(len(left) + 0.5))
        name_ls.append(name)
    return (absscale, relscale, logscale, name_ls)


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


def get_prediction(model, texts, tokenizer, device):
    res = []
    with torch.no_grad():
        for idx, x in enumerate(texts):
            if idx % 1000 == 0:
                print(idx)
            inputs = tokenizer(
                x, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            # perform inference to our model
            outputs = model(**inputs)
            # get output probabilities by doing softmax
            logits = outputs["logits"]
            argmax = logits.argmax(1)
            res.append(argmax)
    return res


def scale_topic(
    dataframe,
    model_transformer,
    text_var,
    group_var,
    guide_var,
    random_state,
    n_neighbors=None,
    batch_size=512,
    max_size=100000,
    guidance_weight=0.5,
    n_components=2,
    use_labels=True,
    auto_k=True,
    use_pca=True,
    **kwargs
):
    enc = OrdinalEncoder()
    frames = []
    pca = PCA(n_components=50, random_state=random_state)
    ## Shuffle to match other types of scaling (by party - election)
    for name, group in dataframe.groupby(group_var):
        frames.append(group)
    df_shuffle = pd.concat(frames).reset_index(drop=True)
    texts = df_shuffle[text_var].tolist()
    col_idx = df_shuffle.columns.get_loc(guide_var)
    document_embed = model_transformer.encode(
        texts, batch_size=batch_size, device="cuda"
    )
    print(f"The shape of this document embeddings is: {document_embed.shape}")
    if auto_k:
        if len(df_shuffle) <= 10000:
            k_recommended = 250
        elif 10000 <= len(df_shuffle) <= 50000:
            k_recommended = 500
        elif len(df_shuffle) >= 50000:
            k_recommended = 750
    else:
        k_recommended = n_neighbors

    if len(df_shuffle) >= max_size:  ## tune max size according to your hardware
        idx = np.random.randint(document_embed.shape[0], size=max_size)
        sample_embed = document_embed[idx, :]
        if use_labels:
            y = enc.fit_transform(
                np.reshape(df_shuffle.iloc[idx, col_idx].tolist(), newshape=(-1, 1))
            )
        else:
            y = None
        embeddings_reduced_s = pca.fit_transform(sample_embed)
        embeddings_reduced_f = pca.fit_transform(document_embed)
        if use_pca: 
            umap_fit = umap.UMAP(
                n_components=n_components,
                n_neighbors=k_recommended,
                n_epochs=200,
                target_weight=guidance_weight,
                low_memory=True,
                verbose=True,
                random_state=random_state,
                **kwargs

            ).fit(embeddings_reduced_s, y=y)
            embeddings = umap_fit.transform(embeddings_reduced_f)
        else:
            umap_fit = umap.UMAP(
            n_components=n_components,
            n_neighbors=k_recommended,
            n_epochs=200,
            target_weight=guidance_weight,
            low_memory=True,
            verbose=True,
            random_state=random_state,
            **kwargs

        ).fit(sample_embed, y=y)
        embeddings = umap_fit.transform(document_embed)
    else:
        if use_labels:
            y = enc.fit_transform(
                np.reshape(df_shuffle.loc[:, guide_var].tolist(), newshape=(-1, 1))
            )
        else:
            y = None
        embeddings_reduced = pca.fit_transform(document_embed)
        if use_pca:
            embeddings = umap.UMAP(
                n_components=n_components,
                n_neighbors=k_recommended,
                n_epochs=200,
                target_weight=guidance_weight,
                low_memory=True,
                verbose=True,
                random_state=random_state,
                **kwargs

            ).fit_transform(embeddings_reduced, y=y)
        else:
            embeddings = umap.UMAP(
            n_components=n_components,
            n_neighbors=k_recommended,
            n_epochs=200,
            target_weight=guidance_weight,
            low_memory=True,
            verbose=True,
            random_state=random_state,
            **kwargs
        ).fit_transform(document_embed, y=y)
        

    return df_shuffle,embeddings,y,document_embed


def train_ae(dataloader, model, optimizer, device, ae_lossf):
    print("")
    print("Training...")

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
        if batch_num % 100 == 0 and batch_num != 0:
            elapsed = format_time(time.time() - t0)
            current = batch_num * len(batch[0])
            train_loss_current = train_loss / batch_num
            print(
                f"loss: {train_loss_current:>7f}  [{current:>5d}/{size:>5d}]. Took {elapsed}"
            )

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Training epoch took: {:}".format(training_time))

def sentiment_code(code):
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

def topic_code(code):
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
