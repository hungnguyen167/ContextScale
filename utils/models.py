import torch.nn as nn
from transformers import XLMRobertaModel
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from gensim.models import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.doc2vec import TaggedDocument

## Main model



class TIPredictWithDualAttention(nn.Module):
    def __init__(self, 
                 roberta_model, 
                 num_sentiments, 
                 num_topics, 
                 lora=False, 
                 dropout=0.1):
        super(TIPredictWithDualAttention, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model)
        roberta_dim = self.roberta.config.hidden_size
        
        if lora:
            lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, use_rslora=True)
            self.roberta = get_peft_model(self.roberta, lora_config)

        self.num_topics = num_topics
        self.num_sentiments = num_sentiments
        hidden_dim = int((roberta_dim + num_topics + num_sentiments) // 2)
        self.dropout = nn.Dropout(dropout)

        # Self-attention for topic and sentiment pathways
        self.self_attn_topic = nn.MultiheadAttention(embed_dim=roberta_dim, num_heads=8, dropout=0.1)
        self.self_attn_sentiment = nn.MultiheadAttention(embed_dim=roberta_dim, num_heads=8, dropout=0.1)
        
        # Cross-attention to allow indirect information flow
        self.cross_attn_topic = nn.MultiheadAttention(embed_dim=roberta_dim, num_heads=8, dropout=0.1)
        self.cross_attn_sentiment = nn.MultiheadAttention(embed_dim=roberta_dim, num_heads=8, dropout=0.1)
        
        # Layer normalization and dropout
        self.layer_norm_topic = nn.LayerNorm(roberta_dim)
        self.layer_norm_sentiment = nn.LayerNorm(roberta_dim)
        
        # Classification Heads
        # Topic Classification Head
        self.mlp_topic = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            self.dropout,  
            nn.Linear(hidden_dim, num_topics)
        )
        
        # Sentiment Classification Head
        self.mlp_sentiment = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            self.dropout,  
            nn.Linear(hidden_dim, num_sentiments)
        )

    def forward(self, input_ids, attention_mask):
        # RoBERTa Encoding
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = roberta_outputs.pooler_output  # (batch_size, hidden_size)

        ### 1. Topic Pathway ###
        # Self-attention in the topic pathway
        self_attn_topic_output, _ = self.self_attn_topic(pooled_output.unsqueeze(0), pooled_output.unsqueeze(0), pooled_output.unsqueeze(0))
        self_attn_topic_output = self.layer_norm_topic(self_attn_topic_output.squeeze(0) + pooled_output)  # Residual connection

        # Cross-attention in the topic pathway (attending to sentiment pathway’s self-attention output)
        cross_attn_topic_output, _ = self.cross_attn_topic(self_attn_topic_output.unsqueeze(0), self_attn_topic_output.unsqueeze(0), self_attn_topic_output.unsqueeze(0))
        cross_attn_topic_output = self.layer_norm_topic(cross_attn_topic_output.squeeze(0) + self_attn_topic_output)

        ### 2. Sentiment Pathway ###
        # Self-attention in the sentiment pathway
        self_attn_sentiment_output, _ = self.self_attn_sentiment(pooled_output.unsqueeze(0), pooled_output.unsqueeze(0), pooled_output.unsqueeze(0))
        self_attn_sentiment_output = self.layer_norm_sentiment(self_attn_sentiment_output.squeeze(0) + pooled_output)  # Residual connection
        
        # Cross-attention in the sentiment pathway (attending to topic pathway’s self-attention output)
        cross_attn_sentiment_output, _ = self.cross_attn_sentiment(self_attn_sentiment_output.unsqueeze(0), cross_attn_topic_output.unsqueeze(0), cross_attn_topic_output.unsqueeze(0))
        cross_attn_sentiment_output = self.layer_norm_sentiment(cross_attn_sentiment_output.squeeze(0) + self_attn_sentiment_output)

        ### Classification Heads ###
        logits_topic = self.mlp_topic(cross_attn_topic_output)
        logits_sentiment = self.mlp_sentiment(cross_attn_sentiment_output)

        return logits_topic, logits_sentiment


    def get_trainable_params(self, return_count=False):
        if return_count:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return total_params
        else:
            print("Trainable Parameters:")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)




class corpusIterator(object):
    def __init__(self, df, bigram=None, trigram=None,text=None,labels=None):
        if bigram: 
            self.bigram = bigram
        else:
            self.bigram = None
        if trigram:
            self.trigram = trigram
        else:
            self.trigram = None
        self.df = df
        self.text = text
        self.labels = labels
    def __iter__(self):
        print('Starting new epoch')
        for  index, row in self.df.iterrows():
            text = row[self.text]
            labels = row[self.labels]
            tokens = text.split()
            if self.bigram and self.trigram:
                self.words = self.trigram[self.bigram[tokens]]
            elif self.bigram and not self.trigram:
                self.words = self.bigram[tokens]
            else:
                self.words = tokens
            yield TaggedDocument(self.words, [labels])
            
class phraseIterator(object):
    def __init__(self, df, text):
        self.df = df
        self.text = text
    def __iter__(self):
        for index, row in self.df.iterrows():
            text = row[self.text]
            yield text.split()