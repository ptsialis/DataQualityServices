import torch
import torch.nn as nn
import torch.nn.functional as F

class FEAT(nn.Module):
    def __init__(self, encoder, hidden_dim=640, temp1=1.0, temp2=1.0, use_cosine=True, dropout_rate=0.5,
                 proto_attn_layers=3, proto_attn_heads=1,
                 aux_transformer_layers=1, aux_transformer_heads=1, aux_transformer_ffn_dim_factor=4):
        super().__init__()
        self.temperature1 = temp1
        self.temperature2 = temp2
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_rate)
        self.use_cosine = use_cosine

        self.prototype_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=proto_attn_heads,batch_first=True, dropout=dropout_rate) for _ in range(proto_attn_layers)])
        self.prototype_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(proto_attn_layers)])

        aux_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=aux_transformer_heads,
            dim_feedforward=hidden_dim * aux_transformer_ffn_dim_factor,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=False
        )
        self.auxiliary_transformer = nn.TransformerEncoder(aux_encoder_layer, num_layers=aux_transformer_layers)
        print(f"\n[FEAT Configuration]")
        print(f"• Backbone output dim: {hidden_dim}")
        print(f"• Proto attention: {proto_attn_layers} layers, {proto_attn_heads} heads")
        print(f"• Aux transformer: {aux_transformer_layers} layers")
        print(f"• Temperatures: main={temp1:.2f}, aux={temp2:.2f}")
        print(f"• Similarity: {'cosine' if use_cosine else 'euclidean'}\n")

    def forward(self, support_x, support_y, query_x, query_y=None, mode='eval'):
        
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)

        unique_classes = torch.unique(support_y)
        num_classes = len(unique_classes)

        initial_prototypes = torch.stack([
            support_emb[support_y == c].mean(0) for c in unique_classes
        ])

        proto_attn_input = initial_prototypes.unsqueeze(0)
        for attn_layer, norm_layer in zip(self.prototype_attention_layers, self.prototype_norm_layers):
            attn_output, _ = attn_layer(proto_attn_input, proto_attn_input, proto_attn_input)
            proto_attn_input = norm_layer(proto_attn_input + self.dropout(attn_output))

        refined_prototypes = proto_attn_input.squeeze(0)

        if self.use_cosine:
            query_norm = F.normalize(query_emb, p=2, dim=1)
            proto_norm = F.normalize(refined_prototypes, p=2, dim=1)
            logits = torch.mm(query_norm, proto_norm.t()) / (self.temperature1 + 1e-8)
        else:
            logits = -torch.cdist(query_emb, refined_prototypes) * self.temperature1

        if mode != 'train':
            return logits

        # --- Training mode only ---
        assert query_y is not None, "query_y must be provided during training"

        aux_input = []
        for c in unique_classes:
            support_c = support_emb[support_y == c]
            query_c = query_emb[query_y == c]
            combined = torch.cat([support_c, query_c], dim=0)
            aux_input.append(combined)

        try:
            aux_tensor = torch.stack(aux_input, dim=0)
        except RuntimeError as e:
            raise RuntimeError(f"Error stacking tensors in FEAT aux path: {[t.shape for t in aux_input]} | {str(e)}")

        transformed = self.auxiliary_transformer(aux_tensor)
        aux_centers = transformed.mean(dim=1)
        flat_samples = transformed.reshape(-1, self.hidden_dim)

        if self.use_cosine:
            s_norm = F.normalize(flat_samples, p=2, dim=1)
            c_norm = F.normalize(aux_centers, p=2, dim=1)
            logits_reg = torch.mm(s_norm, c_norm.t()) / (self.temperature2 + 1e-8)
        else:
            logits_reg = -torch.cdist(flat_samples, aux_centers) * self.temperature2

        expected = num_classes * aux_tensor.size(1)
        assert logits_reg.shape == (expected, num_classes), \
            f"Expected logits_reg shape ({expected}, {num_classes}), got {logits_reg.shape}"

        return logits, logits_reg
