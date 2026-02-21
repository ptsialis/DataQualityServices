#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class FEAT(nn.Module):
    def __init__(self, encoder, hidden_dim= 640, temp1=1.0,temp2= 1.0 , use_cosine=True, dropout_rate=0.5,
                # For prototype attention (aligning with FEAT original for CUB)
                proto_attn_layers=3, proto_attn_heads=1,
                # For auxiliary transformer (aligning with FEAT original for CUB)
                aux_transformer_layers=1, aux_transformer_heads=1, aux_transformer_ffn_dim_factor=4):

        super().__init__()
        self.temperature1 = temp1
        self.temperature2= temp2
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_rate)
        

        assert self.hidden_dim == hidden_dim, \
            f"Hidden dim from encoder ({self.hidden_dim}) does not match args ({hidden_dim})"
        
        self.use_cosine = use_cosine
        self.prototype_attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=2, batch_first=True, dropout=0.1) for _ in range(3)])
        self.prototype_norm_layers = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(3)])


        aux_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=aux_transformer_heads,
            dim_feedforward=self.hidden_dim * aux_transformer_ffn_dim_factor,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=False # Standard: Post-LN. Set True for Pre-LN. FEAT likely uses Post-LN.
        )
        self.auxiliary_transformer = nn.TransformerEncoder(
            encoder_layer=aux_transformer_encoder_layer,
            num_layers=aux_transformer_layers
        )
        
        print(f"FEAT Model Initialized: hidden_dim={self.hidden_dim}, temp1={temp1}, temp2={temp2}, use_cosine={use_cosine}")
        print(f"Prototype Attention: {proto_attn_layers} layers, {proto_attn_heads} heads.")
        print(f"Auxiliary Transformer: {aux_transformer_layers} layers, {aux_transformer_heads} heads, FFN factor {aux_transformer_ffn_dim_factor}.")



    def forward(self, support_x, support_y, query_x, query_y, mode='eval'):
        # Encode all images
        model_device = next(self.encoder.parameters()).device
        support_emb = self.encoder(support_x.to(model_device))
        query_emb = self.encoder(query_x.to(model_device))
        unique_classes = torch.unique(support_y)
        num_classes_episode = len(unique_classes)
        
        initial_prototypes = torch.stack([support_emb[support_y == c].mean(0).contiguous()for c in unique_classes]).contiguous()
        proto_attn_input = initial_prototypes.unsqueeze(0)
        
        #x = prototypes.unsqueeze(0).to(support_x.dtype)
        for attn_layer, norm in zip(self.prototype_attention_layers, self.prototype_norm_layers):
            attn_out, _ = attn_layer(proto_attn_input, proto_attn_input, proto_attn_input)
            proto_attn_input = norm(proto_attn_input + self.dropout(attn_out))  # Residual + norm
        refined_prototypes = proto_attn_input.squeeze(0)                      
        
        # Logits
        if self.use_cosine:
            query_norm = F.normalize(query_emb, p=2, dim=1)
            proto_norm = F.normalize(refined_prototypes, p=2, dim=1)
            logits = torch.mm(query_norm, proto_norm.t()) / max(self.temperature1, 1e-8)
        else:
            logits = -torch.cdist(query_emb, refined_prototypes) * self.temperature1 
            assert logits.shape == (query_x.shape[0], num_classes_episode), f"Shape mismatch for main logits! Expected ({query_x.shape[0]},{num_classes_episode}), got{logits.shape}"

        if mode != 'train':
            return logits
        way = len(unique_classes)
        shot = support_emb.size(0) // num_classes_episode
        num_query_samples_in_episode = query_emb.size(0)
        
        aux_task_input_list = []
        for c_idx, c_label in enumerate(unique_classes):
            support_for_c = support_emb[support_y == c_label]
            query_for_c = query_emb[query_y == c_label] # query_y is the ground truth for query_x
            combined = torch.cat([support_for_c, query_for_c], dim=0)
            aux_task_input_list.append(combined)
        try:
            aux_task_input_tensor = torch.stack(aux_task_input_list, dim=0) 
        except RuntimeError as e:
            print(f"Error stacking for aux task. Shapes per class: {[t.shape for t in aux_task_input_list]}. Error: {e}")
        current_shot = support_emb.shape[0] // num_classes_episode
        current_query_per_class = query_emb.shape[0] // num_classes_episode
        reshaped_support_emb = support_emb.view(num_classes_episode, current_shot, self.hidden_dim)
        reshaped_query_emb = query_emb.view(num_classes_episode, current_query_per_class, self.hidden_dim)
            

        aux_task_input_tensor = torch.cat([reshaped_support_emb, reshaped_query_emb], dim=1)

        transformed_aux_embeddings = self.auxiliary_transformer(aux_task_input_tensor)
        aux_centers = transformed_aux_embeddings.mean(dim=1) # Mean over the (Shot + Query_per_class) dimension
        all_transformed_aux_samples = transformed_aux_embeddings.reshape(-1, self.hidden_dim)

        if self.use_cosine:
            all_transformed_aux_samples_norm = F.normalize(all_transformed_aux_samples, p=2, dim=1)
            aux_centers_norm = F.normalize(aux_centers, p=2, dim=1)
            logits_reg = torch.mm(all_transformed_aux_samples_norm, aux_centers_norm.t()) / max(self.temperature2, 1e-8)
        else:
            logits_reg = -torch.cdist(all_transformed_aux_samples, aux_centers) * self.temperature2
            
        # Expected shape for logits_reg: (Way * (Shot + Query_per_class), Way)
        expected_N_aux = num_classes_episode * (current_shot + current_query_per_class)
        assert logits_reg.shape == (expected_N_aux, num_classes_episode), \
            f"Shape mismatch for regularization logits! Expected ({expected_N_aux}, {num_classes_episode}), got {logits_reg.shape}"
            
        return logits, logits_reg
        
      




