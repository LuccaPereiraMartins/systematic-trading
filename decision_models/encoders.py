"""Full-document FinBERT/BGE heads and LoRA, adapted from Matt's frozen-encoder baseline."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from schemas import LABELS


MODELS = {
    "finbert": ("ProsusAI/finbert", "4556d13015211d73dccd3fdd39d39232506f3e43"),
    "bge": ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
}


class EncoderModel(nn.Module):
    def __init__(self, device="cuda", kind="finbert", adaptation="head", lora_rank=8, checkpoint=None, config=None):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        config = saved["config"] if saved else (config or {})
        if tuple(config.get("labels", LABELS)) != LABELS:
            raise ValueError("Checkpoint label order differs from the benchmark")
        self.training_config = config or None
        self.selected_epoch = saved["state"]["epoch"] if saved else None
        self.device = torch.device(device)
        self.kind = config.get("kind", kind)
        self.adaptation = config.get("adaptation", adaptation)
        self.lora_rank = config.get("lora_rank", lora_rank)
        self.model_id, self.revision = MODELS[self.kind]
        self.model_id = config.get("model", self.model_id)
        self.revision = config.get("revision", self.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.encoder = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        self.encoder.requires_grad_(False)
        if self.adaptation == "lora":
            from peft import LoraConfig, get_peft_model

            self.encoder = get_peft_model(
                self.encoder,
                LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=2 * self.lora_rank,
                    lora_dropout=0.05,
                    target_modules=["query", "value"],
                    bias="none",
                ),
            )
        if self.adaptation != "head":
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        width = self.encoder.config.hidden_size
        self.pool = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        nn.init.zeros_(self.pool[-1].weight)
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.1), nn.Linear(width, len(LABELS)))
        self.cfg = {"max_len": config.get("max_len", 512), "head_max_len": 0}
        self.window_batch = config.get("window_batch", 8)
        self.to(self.device)
        if saved:
            expected = {n for n, p in self.named_parameters() if p.requires_grad}
            if set(saved["weights"]) != expected:
                raise ValueError("Checkpoint trainable parameters do not match the encoder adapter")
            self.load_state_dict(saved["weights"], strict=False)

    def train(self, mode=True):
        super().train(mode)
        if self.adaptation == "head":
            self.encoder.eval()
        return self

    def windows(self, body):
        ids = self.tokenizer.encode(body, add_special_tokens=False, truncation=False, verbose=False)
        # Keep every raw token. No 8-K trimming and no four-window cap from the pilot.
        budget = self.cfg["max_len"] - 2
        return [
            [self.tokenizer.cls_token_id, *ids[i : i + budget], self.tokenizer.sep_token_id]
            for i in range(0, max(1, len(ids)), budget)
        ]

    def features(self, items):
        vectors = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for start in range(0, len(items), self.window_batch):
                rows = [torch.tensor(ids, device=self.device) for ids in items[start : start + self.window_batch]]
                ids = pad_sequence(rows, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                mask = ids != self.tokenizer.pad_token_id
                hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                if self.kind == "bge":
                    pooled = hidden[:, 0]
                else:
                    pooled = (hidden.float() * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
                vectors.append(pooled.float())
        return torch.cat(vectors)

    def forward(self, items):
        vectors = items.to(self.device) if isinstance(items, torch.Tensor) else self.features(items)
        weights = self.pool(vectors).softmax(0)
        return self.classifier((weights * vectors).sum(0)), weights[:, 0]
