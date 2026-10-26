# tokenizer.py


class CharacterTokenizer:
    def fit(self, txt: str, allow_unk_token: bool = True):
        # sorted is necessary to ensure reproducibility
        unique_chars = sorted(list(set(txt)))

        self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        self.id_to_char = {i: char for i, char in enumerate(unique_chars)}

        if allow_unk_token:
            self.char_to_id["UNK"] = len(unique_chars)
            self.id_to_char[len(unique_chars)] = "UNK"

        self.vocab_size = len(list(self.char_to_id.keys()))

    def encode(self, txt: str) -> list[int]:
        token_ids = []
        for char in txt:
            try:
                token_ids.append(self.char_to_id[char])
            except KeyError:
                token_ids.append(self.char_to_id["UNK"])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join([self.id_to_char[id] for id in token_ids])
