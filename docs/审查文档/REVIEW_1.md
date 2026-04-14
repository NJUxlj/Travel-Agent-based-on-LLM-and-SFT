# 代码规范性审查报告

## 审查范围

本次审查覆盖了项目的主要代码文件，包括：

- `src/configs/sft_config.py`
- `src/configs/config.py`
- `src/models/model.py`
- `src/rag/rag.py`
- `src/rag/self_rag.py`
- `src/rag/rag_dispatcher.py`
- `src/finetune/sft_trainer.py`
- `src/finetune/dpo_trainer.py`
- `src/finetune/ppo_trainer.py`
- `src/data/data_processor.py`
- `src/utils/utils.py`
- `main.py`

---

## 发现的问题

### 命名规范

1. **类名与模块名冲突**
   - `src/rag/rag.py` 中同时存在类名 `RAG`（PascalCase），但模块名也是 `rag`，PEP8 建议类名应语义清晰以区分，建议改为如 `BaseRAG`

2. **类型标注严重缺失**
   - `src/models/model.py:43-51` 多个布尔参数缺少类型标注，如 `sft_trainer = None, dpo_trainer = None` 应标注为 `Optional[SFTTrainer]`
   - `src/rag/rag.py:89-98` `RAG.__init__` 大量参数无类型标注
   - `src/rag/rag_dispatcher.py:24` `async def dispatch(self, query:str)` 返回类型缺失，应为 `-> str`

3. **导入路径与实际文件结构不符**
   - `main.py:15` `from src.agents.rag import RAG` 和 `from src.agents.rag import CityRAG`，但 `RAG` 类实际位于 `src/rag/rag.py`，并非 `src/agents/rag.py`
   - `main.py:21` `from src.agents.agent import MyAgent`，但 `src/agents/` 下并无 `agent.py`

4. **常量命名与惯例不符**
   - `src/configs/config.py` 中 `RAG_DATA_PATH`、`SFT_MODEL_PATH`、`DPO_DATA_PATH` 等路径常量使用 SCREAMING_SNAKE_CASE（全大写），但 Python 惯例中只有真正不可变的"常量"才使用全大写，路径常量一般用小写或普通变量命名

5. **函数命名过长**
   - `src/utils/utils.py` 中 `get_model_name_using_model`、`get_base_model_using_model`、`get_classifier_from_model`、`get_hidden_size_using_model` 等函数名过长，虽清晰但略显冗余

---

### 注释问题

1. **docstring 与实际参数不符**
   - `src/models/model.py:53-62` `TravelAgent.__init__` 的 docstring 仅描述了 `model_name` 和 `device_map` 两个参数，但该方法实际有十余个参数（`lora_config`, `sft_trainer`, `use_bnb`, `use_lora`, `use_sft`, `use_dpo`, `use_ppo`, `use_grpo`, `use_api` 等），严重不完整

2. **docstring 位置错误**
   - `src/configs/sft_config.py:41` `"""优化器配置"""` 出现在 `TrainingConfig` 类的属性之间，而非标准的类 docstring 位置（类开头）
   - `src/rag/rag.py:265-268` `chat` 方法的 docstring 使用单引号 `'''simple chat without RAG'''`，且位于方法体内，不是标准的 docstring

3. **中英混杂注释**
   - 几乎所有文件都存在中文注释夹杂英文变量/函数名，例如：
     - `src/models/model.py:24` `"modeling_qwen2 导包出现问题， 应该是transformers版本过低："`
     - `src/finetune/sft_trainer.py:154` `"加载模型和分词器 # 添加LoRA"`
     - `src/finetune/dpo_trainer.py:96` `"计算DPO损失  L = -log(σ(r(x,y_w) - r(x,y_l)))"`
   - 建议统一注释语言

4. **多余/误导性注释**
   - `src/rag/rag.py:133-140` 整块被注释的 `parse_db` 方法（~7行），从未使用
   - `src/models/model.py:137-143` 大段被注释的模型加载代码块
   - `src/rag/self_rag.py:70-90` 大段被注释的 `_init_vector_store` 方法
   - `src/finetune/dpo_trainer.py:464-519` 约55行的注释代码块
   - `src/rag/rag.py:236-243` 注释掉的查询相关代码

5. **未完成的 placeholder 无 TODO 标记**
   - `src/rag/rag_dispatcher.py:40-41` `def rag(self, query:str): pass` 和 `def corrective_rag(self, query:str): pass` 仅有 `pass`，无 `# TODO:` 说明
   - `src/utils/utils.py:554-558` `get_hidden_size_using_config()` 和 `get_hidden_size_by_model_name()` 是空函数（只有 `pass`），无 TODO 说明

6. **docstring 格式混乱**
   - `src/finetune/dpo_trainer.py:107-124` `_get_log_probs` 的 docstring 以 `##Args:` 开头（应为 `Args:`），且 `##Return` 格式错误

---

### 文档问题

1. **README 与代码不一致**
   - README 描述的目录结构中包含 `corrective_rag.py`（"现在还没写好这个，别跑！"），但 `rag_dispatcher.py` 中已引用该模块
   - README 中描述的项目结构与实际文件结构部分不符

2. **硬编码路径问题**
   - `src/configs/config.py` 中大量硬编码绝对路径如 `/root/autodl-tmp/models/...`，不具有可移植性

3. **模块导入路径混乱**
   - `src/rag/rag.py:62` 导入 `from src.configs.config import ...`，应使用相对导入（`from ..configs.config import ...`）
   - `src/finetune/sft_trainer.py:25-26` 使用 `sys.path.append` 动态添加路径来导入模块，不是最佳实践

---

### 格式问题

1. **字符串引号不统一**
   - 代码中混用单引号 `'` 和双引号 `"`，无统一规则
   - `src/finetune/sft_trainer.py` 大量使用单引号，`src/rag/rag.py` 大量使用双引号

2. **空行使用不统一**
   - `src/rag/rag.py` 在类方法之间有的有空行有的没有
   - `src/finetune/dpo_trainer.py` 大段代码之间空行不一致

3. **行长度超限**
   - 多处行的长度远超 79/120 字符限制
   - `src/rag/rag.py:433` 长的 `SYS_PROMPT` 字符串
   - `src/finetune/dpo_trainer.py:464-519` 约55行的注释代码块

4. **文件末尾多余空行**
   - `src/configs/config.py` 末尾有大量空行
   - `src/rag/rag.py` 末尾最后15+行为空

5. **多余导入未清理**
   - `src/utils/utils.py:67` 和 `70` `import os` 和 `import sys` 重复导入
   - `src/finetune/dpo_trainer.py:32-33` 注释掉的 `from trl import DPOTrainer` 和 `import deepspeed`

6. **`if __name__ == '__main__':` 块格式不一致**
   - `src/finetune/sft_trainer.py:452-471` 的 `if __name__` 块中代码未格式化
   - `src/finetune/dpo_trainer.py:527-530` 只有简单两行调用

7. **`src/configs/sft_config.py` 中 docstring 分隔符误用**
   - `# """优化器配置"""` 作为分隔符而非 docstring 本身，这是不恰当的用法

---

## 良好实践

1. **整体项目结构清晰**
   - `src/` 目录按功能模块划分（agents、finetune、rag、models、data、configs），结构合理
   - 每个子模块都有 `__init__.py` 文件，符合 Python 包规范

2. **数据类型使用 dataclass**
   - `src/configs/sft_config.py` 使用 `@dataclass` 定义配置类（`TrainingConfig`、`DataConfig`、`SFTConfig`），比普通类更简洁
   - `src/finetune/dpo_trainer.py` 中的 `DPOTrainingConfig` 使用得当

3. **部分方法有类型注解**
   - `src/rag/self_rag.py` 的方法参数有类型注解
   - `main.py:205` `async def use_rag_dispatcher(rag_type:str="self_rag"):` 使用了类型注解

4. **错误处理**
   - 多处使用 `try/except` 捕获导入错误，避免因依赖缺失导致程序崩溃
   - 如 `main.py:26-28` 的导入错误处理

5. **日志记录**
   - `src/data/data_processor.py` 使用了 `logging` 模块进行日志记录，设置了格式
   - `src/finetune/sft_trainer.py` 使用 `swanlab` 进行实验记录

6. **常量集中管理**
   - `src/configs/config.py` 集中管理所有路径常量和配置常量，避免硬编码分散在多个文件中

7. **跨平台路径处理**
   - 使用了 `os.path.join` 而非字符串拼接处理路径

8. **docstring 基本完整（部分）**
   - `src/finetune/ppo_trainer.py` 的 `PPOTrainer` 类有完整的类 docstring
   - `src/finetune/ppo_trainer.py` 的 `Critic` 类有清晰的注释说明其作用

---

## 总结

该项目整体代码结构清晰，模块划分合理，但在代码规范性方面存在以下主要问题需要改进：

1. **docstring 严重不完整**：多个重要类的 `__init__` 方法 docstring 未列出所有参数，格式也有错误
2. **导入路径混乱**：main.py 中的导入路径与实际文件结构不符，应统一修正
3. **注释代码未清理**：大量被注释的代码块应删除而非保留
4. **dead code 未清理**：多个空 placeholder 函数应使用 `# TODO:` 统一标注
5. **注释语言不统一**：建议统一使用中文或英文注释，避免混杂
6. **代码格式不统一**：存在引号混用、缩进混用（空格/Tab）、空行使用不一致等问题
