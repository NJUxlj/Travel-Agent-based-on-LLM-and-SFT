import yaml
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


def _get_env_or_default(key: str, default: str) -> str:
    """Get environment variable or return default value."""
    return os.getenv(key) or default


BATCH_SIZE = 8
NUM_PROCESSES = 2


OUTPUT_DIR = _get_env_or_default("OUTPUT_DIR", "output/")
MODEL_PATH = _get_env_or_default("MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-0.5B")


REWARD_MODEL_PATH = _get_env_or_default("REWARD_MODEL_PATH", "/root/autodl-tmp/models/reward-model-deberta-v3-large-v2")

EMBEDDING_MODEL_PATH = _get_env_or_default("EMBEDDING_MODEL_PATH", "/root/autodl-tmp/models/all-MiniLM-L6-v2")

EMBEDDING_MODEL_PATH_BPE = _get_env_or_default("EMBEDDING_MODEL_PATH_BPE", "D:\\models\\bge-small-en-v1.5")

SFT_MODEL_NAME = "qwen2_sft"
SFT_MODEL_PATH = os.path.join(OUTPUT_DIR, SFT_MODEL_NAME)


DPO_MODEL_NAME = "qwen2_dpo"
DPO_MODEL_PATH = os.path.join(OUTPUT_DIR, DPO_MODEL_NAME)


PPO_MODEL_NAME = "qwen2_ppo"
PPO_MODEL_PATH = os.path.join(OUTPUT_DIR, PPO_MODEL_NAME)


GRPO_MODEL_NAME = "qwen2_grpo"
GRPO_MODEL_PATH = os.path.join(OUTPUT_DIR, GRPO_MODEL_NAME)


TRPO_MODEL_NAME = "qwen2_trpo"
TRPO_MODEL_PATH = os.path.join(OUTPUT_DIR, TRPO_MODEL_NAME)


SFT_DPO_MODEL_NAME = "qwen2_sft_dpo"
SFT_DPO_MODEL_PATH = os.path.join(OUTPUT_DIR, SFT_DPO_MODEL_NAME)



DATA_PATH = _get_env_or_default("DATA_PATH", "src/data/travel_qa")
RAG_DATA_PATH = _get_env_or_default("RAG_DATA_PATH", "src/data/crosswoz-sft")

DPO_DATA_PATH = _get_env_or_default("DPO_DATA_PATH", "/root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/Human-Like-DPO-Dataset")

CACHED_SFT_DATA_PATH = _get_env_or_default("CACHED_SFT_DATA_PATH", "/root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/sft_data_cached")
CACHED_DPO_DATA_PATH = _get_env_or_default("CACHED_DPO_DATA_PATH", "/root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/dpo_data_cached")
CACHED_GRPO_DATA_PATH = _get_env_or_default("CACHED_GRPO_DATA_PATH", "/root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/grpo_data_cached")
CACHED_PPO_DATA_PATH = _get_env_or_default("CACHED_PPO_DATA_PATH", "/root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/ppo_data_cached")

DEEPSPEED_CONFIG_PATH = _get_env_or_default("DEEPSPEED_CONFIG_PATH", "src/configs/ds_config.json")

# Model configuration
MODEL_CONFIG = {
    "model": {
        "name": MODEL_PATH,
        "type": "qwen2",
        "trust_remote_code": True
    }
}

PDF_FOLDER_PATH = _get_env_or_default("PDF_FOLDER_PATH", "src/agents/travel_knowledge/tour_pdfs")
PAGE_FOLDER_PATH = _get_env_or_default("PAGE_FOLDER_PATH", "src/agents/travel_knowledge/tour_pages")


