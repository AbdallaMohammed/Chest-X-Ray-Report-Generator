import torch
import utils


def _add_to_dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value

    return dictionary


class Vocab:
    def __init__(self, captions):
        self.caps = captions
        self.tokenizer = utils.load_bert_tokenizer()
        self.tokens_count = {}

        self.max_len = self._get_sentence_max_len()

        self.ids_to_tokens, self.tokens_to_ids = self._make_vocab()
        self.ids_to_tokens_adj, self.ids_to_ids_adj, self.ids_adj_to_ids = self._make_vocab_adj()

    def __len__(self):
        return len(self.ids_to_tokens)

    def _make_vocab(self):
        ids_to_tokens = {}
        tokens_to_ids = {}

        for cap in self.caps:
            encoded_dict = self.tokenizer(
                cap,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )

            for idx in encoded_dict['input_ids'][0]:
                idx_int = int(idx)
                token = self.tokenizer.convert_ids_to_tokens(idx_int)

                ids_to_tokens = _add_to_dict(
                    dictionary=ids_to_tokens,
                    key=idx_int,
                    value=token
                )

                tokens_to_ids = _add_to_dict(
                    dictionary=tokens_to_ids,
                    key=token,
                    value=idx_int
                )

                if token in self.tokens_count.keys():
                    self.tokens_count[token] += 1
                else:
                    self.tokens_count[token] = 1

        return ids_to_tokens, tokens_to_ids

    def _make_vocab_adj(self):
        ids_to_tokens_adj = {}
        ids_to_ids_adj = {}
        ids_adj_to_adj = {}

        for i, id_token in enumerate(self.ids_to_tokens.items()):
            id_, token = id_token

            ids_to_tokens_adj[i] = token
            ids_to_ids_adj[id_] = i
            ids_adj_to_adj[i] = id_

        return ids_to_tokens_adj, ids_to_ids_adj, ids_adj_to_adj

    def _get_sentence_max_len(self):
        max_len = -1

        for cap in self.caps:
            curr_len = len(self.tokenizer.tokenize(cap))
            
            if curr_len > max_len:
                max_len = curr_len

        return max_len

    def map_to_trainset_ids(self, indices_tensor):
        indices_tensor = torch.LongTensor([self.ids_to_ids_adj[int(elem)] for elem in indices_tensor])
        
        return indices_tensor
    
    def map_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)
