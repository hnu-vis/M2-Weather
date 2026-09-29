"""Narrow compatibility helpers for remote forecasting model code."""


def patch_dynamic_cache_seen_tokens():
    """Restore the read-only cache attribute expected by older remote code."""
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda cache: cache.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda cache: None
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda cache, new_seq_length, layer_idx=0: cache.get_seq_length(layer_idx)
        )


def patch_legacy_generation_methods(model):
    """Supply generation helpers removed after the remote code was published."""
    if not hasattr(model, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(
            self, outputs, standardize_cache_format=False
        ):
            for name in ("past_key_values", "mems", "past_buckets_states"):
                value = getattr(outputs, name, None)
                if value is not None:
                    return value
            return None

        model.__class__._extract_past_from_model_output = (
            _extract_past_from_model_output
        )

    model_class = model.__class__
    if (
        "_sample" not in model_class.__dict__
        and hasattr(model, "_greedy_search")
    ):
        def _sample(
            self, input_ids, logits_processor, stopping_criteria,
            generation_config, synced_gpus=False, streamer=None, **model_kwargs
        ):
            # Newer transformers eagerly creates an empty DynamicCache before
            # entering _sample.  Sundial's legacy _greedy_search uses a None
            # cache to identify the first decoding step and initialize its
            # generated-token accumulator.  Let the first forward pass create
            # the cache, matching the transformers version it was written for.
            model_kwargs["past_key_values"] = None
            return self._greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_attentions=generation_config.output_attentions,
                output_hidden_states=generation_config.output_hidden_states,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        model_class._sample = _sample
