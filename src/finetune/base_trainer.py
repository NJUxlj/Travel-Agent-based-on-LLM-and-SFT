"""Base trainer class with common functionality to reduce code duplication across trainers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import copy

from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


@dataclass
class BNBConfig:
    """BitsAndBytes quantization configuration."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True

    def to_bitsandbytes_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )


def create_default_bnb_config() -> BitsAndBytesConfig:
    """Create default BitsAndBytes configuration for 4-bit quantization."""
    return BNBConfig().to_bitsandbytes_config()


def create_default_lora_config() -> LoraConfig:
    """Create default LoRA configuration."""
    return LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )


def init_model_and_tokenizer(
    model_name: str,
    is_quantized: bool = False,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    device_map: str = "auto"
) -> Tuple[Qwen2ForCausalLM, AutoTokenizer]:
    """
    Initialize model and tokenizer with common configuration.

    Args:
        model_name: Path or name of the model
        is_quantized: Whether to use 4-bit quantization
        bnb_config: BitsAndBytes configuration (uses default if None)
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    if bnb_config is None and is_quantized:
        bnb_config = create_default_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if is_quantized else None,
        device_map=device_map,
        trust_remote_code=True
    )
    return model, tokenizer


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """Create a deep copy of a model."""
    return copy.deepcopy(model)


class BaseTrainerWrapper(ABC):
    """
    Abstract base class for trainer wrappers to reduce code duplication.

    Provides common methods for model/tokenizer initialization,
    LoRA configuration, and other shared functionality.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        is_peft: bool = False,
        peft_config: Optional[LoraConfig] = None,
        is_quantized: bool = False,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        max_seq_length: int = 1024,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.is_peft = is_peft
        self.peft_config = peft_config or create_default_lora_config()
        self.is_quantized = is_quantized
        self.bnb_config = bnb_config or (create_default_bnb_config() if is_quantized else None)
        self.max_seq_length = max_seq_length

        # These will be set by _init_model_and_tokenizer
        self.model = None
        self.tokenizer = None
        self.device = None

    @abstractmethod
    def _init_model_and_tokenizer(self) -> Tuple[Qwen2ForCausalLM, AutoTokenizer]:
        """
        Initialize model and tokenizer. Subclasses should override this
        if they need custom initialization logic.
        """
        model, tokenizer = init_model_and_tokenizer(
            self.model_name,
            self.is_quantized,
            self.bnb_config
        )
        return model, tokenizer

    def _setup_model(self):
        """Common model setup logic."""
        self.model, self.tokenizer = self._init_model_and_tokenizer()
        self.device = self.model.device

        if self.is_peft:
            self.model = get_peft_model(self.model, self.peft_config)

    def _create_ref_model(self) -> Qwen2ForCausalLM:
        """
        Create a reference model by deep copying and freezing the main model.
        """
        ref_model = clone_model(self.model)
        ref_model.eval()
        ref_model.requires_grad_(False)
        return ref_model
