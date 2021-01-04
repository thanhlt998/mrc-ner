import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from fastai.basics import L, params

from models.classifier import MultiNonLinearClassifier, SingleLinearClassifier


class PhoBertQueryNER(RobertaPreTrainedModel):
    def __init__(self, config):
        super(PhoBertQueryNER, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.sep_token_id = config.sep_token_id
        self.pad_token_id = config.pad_token_id

        # self.start_outputs = nn.Linear(config.hidden_size, 2)
        # self.end_outputs = nn.Linear(config.hidden_size, 2)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout)
        # self.span_embedding = SingleLinearClassifier(config.hidden_size * 2, 1)

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        bert_outputs = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits


def splitter(m: PhoBertQueryNER):

    groups = [
        m.roberta.embeddings,
        *[m.roberta.encoder.layer[i] for i in range(m.config.num_hidden_layers)],
        nn.ModuleList([
            m.start_outputs,
            m.end_outputs,
            m.span_embedding,
        ])
    ]
    groups = L(groups)
    return groups.map(params)
