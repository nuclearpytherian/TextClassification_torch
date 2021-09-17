import os

class Series_Tokenizer:

    def __init__(self, mode):
        self.mode = mode
        if mode == "Mecab":
            from konlpy.tag import Mecab
            if os.path.isdir('mecab'):
                mecab_dic_path = "mecab\mecab-ko-dic"
                m = Mecab(mecab_dic_path)
                self.tokenizer = m.morphs
            else:
                print("No mecab dic data exists.")

        elif mode == "kobert_tokenizer":
            """
            SAME TO:
            from pytorch_transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            tokens = tokenizer.tokenize(sentence)
            indexes = tokenizer.convert_tokens_to_ids(tokens)
            """
            from kobert_tokenizer.kobert_tokenizer import KoBERTTokenizer
            tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1',
                                                        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6,
                                                                         'enable_sampling': True})
            self.tokenizer = tokenizer._tokenize
            self.encode = tokenizer.encode

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


if __name__=="__main__":
    t = Series_Tokenizer(mode='kobert_tokenizer')
    sentence = "심심이를 끝내러 왔다."
    print(t(sentence))
    print(t.encode(sentence))