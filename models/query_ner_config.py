# encoding: utf-8


from transformers import BertConfig, RobertaConfig


class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)


class PhobertQueryNerConfig(RobertaConfig):
    def __init__(self, **kwargs):
        super(PhobertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get('mrc_dropout', 0.1)
