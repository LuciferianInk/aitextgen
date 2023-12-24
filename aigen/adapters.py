from peft import (
    AdaLoraConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
)


def get_peft_config(peft_type, kwargs):
    if peft_type == "lora":
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=kwargs.get("r", 4),
            lora_alpha=kwargs.get("alpha", 16),
            lora_dropout=kwargs.get("dropout", 0.1),
            bias=kwargs.get("bias", "none"),
            target_modules=kwargs.get("target_modules", None),
            rank_pattern=kwargs.get("rank_pattern", {}),
            alpha_pattern=kwargs.get("alpha_pattern", {}),
            modules_to_save=kwargs.get("modules_to_save", None),
        )
    elif peft_type == "adalora":
        peft_config = AdaLoraConfig(
            task_type="CAUSAL_LM",
            r=kwargs.get("r", 4),
            init_r=kwargs.get("init_r", 12),
            target_r=kwargs.get("target_r", 8),
            lora_alpha=kwargs.get("alpha", 16),
            lora_dropout=kwargs.get("dropout", 0.1),
            bias=kwargs.get("bias", "none"),
            target_modules=kwargs.get("target_modules", None),
            rank_pattern=kwargs.get("rank_pattern", {}),
            alpha_pattern=kwargs.get("alpha_pattern", {}),
            modules_to_save=kwargs.get("modules_to_save", None),
        )
    elif peft_type == "ia3":
        peft_config = IA3Config(
            task_type="CAUSAL_LM",
            target_modules=kwargs.get("target_modules", None),
            feedforward_modules=kwargs.get("feedforward_modules", None),
        )
    elif peft_type == "loha":
        peft_config = LoHaConfig(
            task_type="CAUSAL_LM",
            r=kwargs.get("r", 8),
            alpha=kwargs.get("alpha", 8),
            rank_dropout=kwargs.get("dropout", 0.0),
            module_dropout=kwargs.get("dropout", 0.0),
            use_effective_conv2d=False,
            target_modules=kwargs.get("target_modules", None),
            rank_pattern=kwargs.get("rank_pattern", {}),
            alpha_pattern=kwargs.get("alpha_pattern", {}),
            modules_to_save=kwargs.get("modules_to_save", None),
        )
    elif peft_type == "lokr":
        peft_config = LoKrConfig(
            task_type="CAUSAL_LM",
            r=kwargs.get("r", 8),
            alpha=kwargs.get("alpha", 8),
            rank_dropout=kwargs.get("dropout", 0.0),
            module_dropout=kwargs.get("dropout", 0.0),
            use_effective_conv2d=False,
            decompose_both=True,
            decompose_factor=-1,
            target_modules=kwargs.get("target_modules", None),
            rank_pattern=kwargs.get("rank_pattern", {}),
            alpha_pattern=kwargs.get("alpha_pattern", {}),
            modules_to_save=kwargs.get("modules_to_save", None),
        )
    elif peft_type == "prompt":
        # if use_petals:
        #     tuning_mode = "ptune"
        #     pre_seq_len = kwargs.get("num_virtual_tokens")
        #     output_dir = "/data/embeddings/" + focus
        # else:
        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=kwargs.get("num_virtual_tokens", 24),
        )
    elif peft_type == "prefix":
        # if use_petals:
        #     tuning_mode = "deep_ptune"
        #     pre_seq_len = kwargs.get("num_virtual_tokens")
        #     output_dir = "/data/embeddings/" + focus
        # else:
        peft_config = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=kwargs.get("num_virtual_tokens", 24),
        )
    else:
        pass
    #     output_dir = "/data/models/" + focus

    return peft_config
