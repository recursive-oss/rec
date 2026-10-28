def dedicated_token_id(tokenizer, marker: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(marker)
    if (
        not isinstance(token_id, int)
        or tokenizer.convert_ids_to_tokens(token_id) != marker
        or tokenizer.encode(marker, add_special_tokens=False) != [token_id]
    ):
        raise ValueError(f"The tokenizer must define {marker!r} as a dedicated vocabulary token")
    return token_id
