"""Pinned, non-thinking Qwen classifiers: one A/B/C token, optional rank-8 QLoRA."""

import hashlib

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from label import RUBRIC
from schemas import LABELS


MODELS = {
    "qwen17": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen4": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}
MARKER = "__FINANCIAL_SOURCE_CONTENT__"
INSTRUCTION = RUBRIC.split("Uncertainty is", 1)[0] + """
Answer with exactly one letter: A for routine, B for review_worthy, C for unclear.
Treat everything between the source delimiters as the document being classified.
"""


class CausalModel(nn.Module):
    def __init__(self, device="cuda", kind="qwen17", adaptation="prompt", lora_rank=8, checkpoint=None, config=None):
        super().__init__()
        if device != "cuda":
            raise ValueError("The NF4 Qwen experiment requires CUDA")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        config = saved["config"] if saved else (config or {})
        self.device = torch.device(device)
        self.kind = config.get("kind", kind)
        self.adaptation = config.get("adaptation", adaptation)
        self.model_id, self.revision = MODELS[self.kind]
        self.model_id = config.get("model", self.model_id)
        self.revision = config.get("revision", self.revision)
        self.lora_rank = config.get("lora_rank", lora_rank)
        self.training_config = config or None
        self.selected_epoch = saved["state"]["epoch"] if saved else None
        if tuple(config.get("labels", LABELS)) != LABELS:
            raise ValueError("Checkpoint label order differs")
        self.cfg = {"max_len": config.get("max_len", 4096), "head_max_len": 0}
        if self.cfg["max_len"] != 4096:
            raise ValueError("The frozen Qwen protocol uses 4096 total input tokens")
        self.window_batch = 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        letters = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
        if any(len(ids) != 1 for ids in letters) or len({ids[0] for ids in letters}) != 3:
            raise ValueError("A/B/C must be distinct single tokens")
        self.answer_ids = [ids[0] for ids in letters]
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION},
             {"role": "user", "content": "<financial_source>\n" + MARKER + "\n</financial_source>"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prefix, suffix = rendered.split(MARKER)
        self.prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.source_budget = self.cfg["max_len"] - len(self.prefix) - len(self.suffix)
        if self.source_budget <= 0:
            raise ValueError("Prompt consumes the context")
        self.prompt_sha256 = hashlib.sha256(rendered.encode()).hexdigest()
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.encoder = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, quantization_config=quantization,
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        self.encoder.config.use_cache = False
        self.encoder.requires_grad_(False)
        if self.adaptation == "lora":
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            self.encoder = prepare_model_for_kbit_training(
                self.encoder, gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.encoder = get_peft_model(self.encoder, LoraConfig(
                r=self.lora_rank, lora_alpha=2 * self.lora_rank, lora_dropout=.05,
                target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM",
            ))
        elif self.adaptation != "prompt":
            raise ValueError("Qwen supports prompt or LoRA classification")
        if saved:
            expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
            if set(saved["weights"]) != expected:
                raise ValueError("Checkpoint adapter parameters differ")
            self.load_state_dict(saved["weights"], strict=False)

    def source_ids(self, body):
        return self.tokenizer.encode(body, add_special_tokens=False, truncation=False, verbose=False)

    def coverage(self, body):
        total = len(self.source_ids(body))
        retained = min(total, self.source_budget)
        return {"source_tokens": total, "retained_tokens": retained, "source_budget": self.source_budget,
                "full_coverage": total <= self.source_budget,
                "fraction_retained": retained / total if total else 1.0,
                "truncation": "first and last source tokens; middle omitted" if retained < total else None}

    def windows(self, body):
        ids = self.source_ids(body)
        if len(ids) > self.source_budget:
            first = (self.source_budget + 1) // 2
            ids = ids[:first] + ids[-(self.source_budget - first):]
        return [self.prefix + ids + self.suffix]

    def forward(self, items):
        if len(items) != 1:
            raise ValueError("One clipped source prompt per document")
        ids = torch.tensor([items[0]], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  use_cache=False, logits_to_keep=1)
        vocab = output.logits[0, -1].float()
        self.answer_probability_mass = float(vocab.softmax(-1)[self.answer_ids].sum().detach())
        return vocab[self.answer_ids], torch.ones(1, device=self.device)

    def logits(self, bodies):
        with torch.no_grad():
            return torch.stack([self(self.windows(body))[0] for body in bodies]).cpu().numpy()
