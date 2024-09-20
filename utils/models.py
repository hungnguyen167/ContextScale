import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.modeling_utils import PreTrainedModel
from peft import LoraConfig, get_peft_model

## Main model

class TIPredictWithAttention(PreTrainedModel):
    def __init__(self, config, roberta_model, num_lrs, num_topics=None, 
                 num_party_groups=None, lora=False, dropout=0.1, hidden_dim=None, emb_dim=None):
        super(TIPredictWithAttention, self).__init__(config)
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model)
        
        # Optional: LoRA adapter
        if lora:
            lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, use_rslora=True)
            self.roberta = get_peft_model(self.roberta, lora_config)

        roberta_dim = config.hidden_size

        self.num_topics = num_topics
        self.num_lrs = num_lrs
        self.num_party_groups = num_party_groups
        if hidden_dim is None:
            hidden_dim = (roberta_dim + num_topics + num_lrs) // 2

        self.dropout = nn.Dropout(dropout)

        # Group Embeddings
        if num_topics:
            self.topic_embedding = nn.Embedding(num_topics, emb_dim)
        if num_party_groups:
            self.party_group_embedding = nn.Embedding(num_party_groups, emb_dim)

        # Multi-Headed Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, dropout=0.1)
        self.attn_dropout = nn.Dropout(0.1)
        self.attn_layer_norm = nn.LayerNorm(roberta_dim)

        # Classification Heads
        # Topic Classification Head
        self.mlp_topic = nn.Sequential(
            nn.Linear(roberta_dim + emb_dim if num_party_groups else roberta_dim, hidden_dim),  # pooled_output + party_group_emb concatenated
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            self.dropout,  
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            self.dropout,  
            nn.Linear(hidden_dim, num_topics)
        )
        
        # Ideology Classification Head
        self.mlp_ideology = nn.Sequential(
            nn.Linear(roberta_dim + emb_dim if num_topics else roberta_dim, hidden_dim),  # pooled_output + attn_output concatenated
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            self.dropout, 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            self.dropout, 
            nn.Linear(hidden_dim, num_lrs)
        )

    def forward(self, input_ids, attention_mask, topic_labels=None, party_group_labels=None):
        # RoBERTa Encoding
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = roberta_outputs.pooler_output  # (batch_size, hidden_size)



        ### 1. Topic Prediction Pathway ###
            # Combine pooled_output with party_group_emb (if available) for topic prediction
        if self.num_party_groups and party_group_labels is not None:
            party_group_emb = self.party_group_embedding(party_group_labels)  # (batch_size, emb_size)
            topic_input = torch.cat([pooled_output, party_group_emb], dim=-1)  # (batch_size, hidden_size + emb_size)
        else:
            # If no party_group, just use pooled_output for topic prediction
            topic_input = pooled_output  # (batch_size, hidden_size)
        logits_topic = self.mlp_topic(topic_input)  # (batch_size, num_topics)

        ### 2. Ideology Prediction Pathway ###
        # Case 1: Both topic and party-group are available
        if self.num_topics and topic_labels is not None and self.num_party_groups and party_group_labels is not None:
            topic_emb = self.topic_embedding(topic_labels)  # (batch_size, emb_size)
            party_group_emb = self.party_group_embedding(party_group_labels)  # (batch_size, emb_size)
            combined_emb = topic_emb * party_group_emb  # (batch_size, emb_size)

            # Reshape for MultiheadAttention: (seq_len=1, batch_size, emb_size)
            combined_emb = combined_emb.unsqueeze(0)  # (1, batch_size, emb_size)

            # Apply Multi-Headed Self-Attention
            attn_output, _ = self.multihead_attn(combined_emb, combined_emb, combined_emb)
            attn_output = self.attn_dropout(attn_output)  # (1, batch_size, emb_size)
            attn_output = self.attn_layer_norm(attn_output + combined_emb)  # Residual connection
            attn_output = attn_output.squeeze(0)  # (batch_size, emb_size)

            # Combine pooled_output with attention output for ideology prediction
            ideology_input = torch.cat([pooled_output, attn_output], dim=-1)  # (batch_size, hidden_size + emb_size)

        # Case 2: Only topic is available (use topic embedding for ideology prediction)
        elif self.num_topics and topic_labels is not None:
            topic_emb = self.topic_embedding(topic_labels)  # (batch_size, emb_size)
            ideology_input = torch.cat([pooled_output, topic_emb], dim=-1)  # (batch_size, hidden_size + emb_size)

        # Case 3: No topic or party-group (only use pooled_output for ideology prediction)
        else:
            ideology_input = pooled_output  # replicate pooled_output for concatenation

        logits_ideology = self.mlp_ideology(ideology_input) # (batch_size, num_lrs)

        return logits_topic, logits_ideology

    def get_trainable_params(self, return_count=False):
        if return_count:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return total_params
        else:
            print("Trainable Parameters:")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)