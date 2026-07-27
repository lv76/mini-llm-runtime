import regex as re


PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def split_by_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]

    segments = []
    i = 0
    last_end = 0

    special_tokens_sorted = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    while i < len(text):
        matched = False

        for special_token in special_tokens_sorted:
            if text.startswith(special_token, i):
                if i > last_end:
                    segments.append(text[last_end:i])

                segments.append(special_token)
                i += len(special_token)
                last_end = i
                matched = True
                break

        if not matched:
            i += 1

    if last_end < len(text):
        segments.append(text[last_end:])

    return segments


class Tokenizer:
    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None,
    ) -> None:
        super().__init__()

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.byte_to_id = {
            bytes_value: token_id
            for token_id, bytes_value in self.vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    def _apply_bpe(self, token_list):
        token_list = tuple(token_list)

        while True:
            best_pair = None
            best_rank = None

            for i in range(len(token_list) - 1):
                pair = (token_list[i], token_list[i + 1])
                rank = self.merge_ranks.get(pair)

                if rank is not None and (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_token_list = []
            i = 0

            while i < len(token_list):
                if (
                    i < len(token_list) - 1
                    and token_list[i] == best_pair[0]
                    and token_list[i + 1] == best_pair[1]
                ):
                    new_token_list.append(
                        token_list[i] + token_list[i + 1]
                    )
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1

            token_list = tuple(new_token_list)

        return list(token_list)

    def encode(self, text):
        if text == "":
            return []

        final_ids = []

        segments = split_by_special_tokens(
            text,
            self.special_tokens,
        )

        for segment in segments:
            if segment == "":
                continue

            if segment in self.special_tokens:
                final_ids.append(
                    self.byte_to_id[segment.encode("utf-8")]
                )
                continue

            for match in re.finditer(PAT, segment):
                piece = match.group(0)

                token_list = [
                    bytes([byte])
                    for byte in piece.encode("utf-8")
                ]

                if len(token_list) == 1:
                    final_ids.append(
                        self.byte_to_id[token_list[0]]
                    )
                    continue

                token_list = self._apply_bpe(token_list)

                final_ids.extend(
                    self.byte_to_id[token]
                    for token in token_list
                )

        return final_ids

    def decode(self, final_ids):
        if len(final_ids) == 0:
            return ""

        bytes_pieces = [
            self.vocab[token_id]
            for token_id in final_ids
        ]

        bytes_sequence = b"".join(bytes_pieces)

        return bytes_sequence.decode(
            "utf-8",
            errors="replace",
        )

    def encode_iterable(self, iterable):
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id
