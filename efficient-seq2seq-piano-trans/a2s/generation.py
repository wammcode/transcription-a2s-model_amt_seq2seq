"""Task-neutral AR decoding through the original decoder and its KV cache."""
import torch


@torch.no_grad()
def generate(model, features, max_tokens, encoder_lengths=None):
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if model.training:
        raise ValueError("Call model.eval() before generation")
    if max_tokens > model.config.max_decode_length:
        raise ValueError("Requested generation exceeds checkpoint positional capacity")
    model.decoder.initialize_decoder_cache()
    encoded = model.encode(features, enable_dropout=False, encoder_lengths=encoder_lengths)
    memory = model.prepare_score_memory(encoded, encoder_lengths)
    batch = features.size(0)
    current = torch.full((batch, 1), model.bos_token, dtype=torch.long, device=features.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=features.device)
    result = []
    # Pooling, positions, self/cross attention and output projection remain in T5/Decoder.
    for position in range(max_tokens):
        logits = model.decode(encoded, encoded, current, torch.ones_like(current),
                              decoder_positions=torch.full_like(current, position),
                              enable_dropout=False, decode=True, score_memory=memory)["decoder_outputs"][:, -1].clone()
        logits[:, [model.pad_token, model.bos_token]] = -torch.inf
        token = logits.argmax(-1)
        token = torch.where(finished, model.pad_token, token)
        result.append(token)
        finished |= token == model.eos_token
        if finished.all():
            break
        current = torch.where(finished, model.eos_token, token)[:, None]
    return torch.stack(result, dim=1)
