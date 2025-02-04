import torch.nn as nn
from transformers import XLMRobertaModel
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from gensim.models import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.doc2vec import TaggedDocument

## Main model

import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class ContextScalePrediction(nn.Module):
    def __init__(self, 
                 roberta_model, 
                 num_sentiments, 
                 num_topics, 
                 lora=False, 
                 dropout=0.1, 
                 use_shared_attention=False, 
                 use_dynamic_gating=False, 
                 use_co_attention=False, 
                 use_simple_flow=False, 
                 use_hierarchical_interaction=False):
        super(ContextScalePrediction, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model)
        roberta_dim = self.roberta.config.hidden_size

        if lora:
            lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, use_rslora=True)
            self.roberta = get_peft_model(self.roberta, lora_config)

        self.num_topics = num_topics
        self.num_sentiments = num_sentiments
        hidden_dim = int(roberta_dim // 2)
        self.dropout = nn.Dropout(dropout)

        # Toggleable components
        self.use_shared_attention = use_shared_attention
        self.use_dynamic_gating = use_dynamic_gating
        self.use_co_attention = use_co_attention
        self.use_simple_flow = use_simple_flow
        self.use_hierarchical_interaction = use_hierarchical_interaction

        # Shared intermediate state
        self.intermediate = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            self.dropout
        )

        # Task-specific intermediate layers
        self.intermediate_topic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            self.dropout
        )
        
        self.intermediate_sentiment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            self.dropout
        )

        # Shared Attention Mechanism
        if self.use_shared_attention:
            self.shared_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        
        # Dynamic Gating
        if self.use_dynamic_gating:
            self.topic_gate = nn.Linear(hidden_dim, hidden_dim)
            self.sentiment_gate = nn.Linear(hidden_dim, hidden_dim)

    
        # Classification Heads
        self.topic = nn.Linear(hidden_dim, num_topics)
        self.sentiment = nn.Linear(hidden_dim, num_sentiments)

        # Regularization after attention
        self.attention_regularizer = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        mean_outputs = roberta_outputs.last_hidden_state.mean(dim=1)
        shared_output = self.intermediate(mean_outputs)

        topic_intermediate = self.intermediate_topic(shared_output)
        sentiment_intermediate = self.intermediate_sentiment(shared_output)

        # Apply shared attention
        if self.use_shared_attention:
            combined_representation = torch.stack([topic_intermediate, sentiment_intermediate], dim=1).transpose(0,1)
            combined_representation, _ = self.shared_attention(combined_representation, combined_representation, combined_representation)
            combined_representation = combined_representation.transpose(0, 1)
            topic_intermediate = combined_representation[:, 0]
            sentiment_intermediate = combined_representation[:, 1]
            topic_intermediate = self.attention_regularizer(topic_intermediate)
            sentiment_intermediate = self.attention_regularizer(sentiment_intermediate)

        # Apply dynamic gating
        if self.use_dynamic_gating:
            topic_gate_weights = torch.sigmoid(self.topic_gate(topic_intermediate))
            sentiment_gate_weights = torch.sigmoid(self.sentiment_gate(sentiment_intermediate))
            topic_intermediate = topic_intermediate + topic_gate_weights * sentiment_intermediate
            sentiment_intermediate = sentiment_intermediate + sentiment_gate_weights * topic_intermediate

        # Use simple flow of information

        if self.use_simple_flow:
            topic_intermediate += 0.1*sentiment_intermediate.detach()
            sentiment_intermediate += 0.1*topic_intermediate.detach()

        logits_topic = self.topic(topic_intermediate)
        logits_sentiment = self.sentiment(sentiment_intermediate)

        return {
            'logits_topic': logits_topic,
            'logits_sentiment': logits_sentiment
        }





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
        for  _, row in self.df.iterrows():
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
        for _, row in self.df.iterrows():
            text = row[self.text]
            yield text.split()