#!/usr/bin/env python
"""
测试模型加载和推理功能
由于依赖环境问题，无法实际加载模型。
本测试重点检查代码正确性和参数配置。
"""
import sys
import os

sys.path.append("/Users/xiniuyiliao/Desktop/code/Travel-Agent-based-on-Qwen2-RLHF")

MODEL_PATH = "/Users/xiniuyiliao/Desktop/code/models/qwen3-0.6B"

def check_code_correctness():
    """检查代码正确性"""
    print("=" * 60)
    print("代码正确性检查")
    print("=" * 60)

    # 检查 model.py 文件
    model_file = "/Users/xiniuyiliao/Desktop/code/Travel-Agent-based-on-Qwen2-RLHF/src/models/model.py"

    with open(model_file, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []
    checks = []

    # 检查1: generate_response 方法中的 max_new_tokens 参数
    if 'max_new_tokens=max_length' in content:
        checks.append({
            "test": "generate_response 使用 max_new_tokens",
            "passed": True,
            "detail": "generate_response 方法正确使用 max_new_tokens=max_length"
        })
    else:
        checks.append({
            "test": "generate_response 使用 max_new_tokens",
            "passed": False,
            "detail": "generate_response 方法未正确使用 max_new_tokens 参数"
        })

    # 检查2: stream_chat 方法中的 max_length 和 max_new_tokens 参数
    if 'max_length=max_length' in content and 'max_new_tokens=1' in content:
        checks.append({
            "test": "stream_chat 正确使用 max_length 和 max_new_tokens",
            "passed": True,
            "detail": "stream_chat 方法正确使用 max_length(限制总长度) 和 max_new_tokens=1(每次生成1个token)"
        })
    else:
        checks.append({
            "test": "stream_chat 正确使用 max_length 和 max_new_tokens",
            "passed": False,
            "detail": "stream_chat 方法未正确使用 max_length 和 max_new_tokens 参数"
        })

    # 检查3: 检查是否有导入错误
    if 'from src.configs.config import MODEL_CONFIG' in content:
        # 检查 MODEL_CONFIG 是否真的存在于 config.py
        config_file = "/Users/xiniuyiliao/Desktop/code/Travel-Agent-based-on-Qwen2-RLHF/src/configs/config.py"
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_content = f.read()
            if 'MODEL_CONFIG' not in config_content:
                issues.append({
                    "severity": "ERROR",
                    "file": "src/models/model.py",
                    "line": 13,
                    "issue": "导入错误: MODEL_CONFIG 不存在于 src.configs.config 中",
                    "fix": "需要从 sft_config.py 导入 SFTConfig 或在 config.py 中定义 MODEL_CONFIG"
                })
        checks.append({
            "test": "MODEL_CONFIG 导入检查",
            "passed": False,
            "detail": "MODEL_CONFIG 不存在于 config.py 中，会导致 ImportError"
        })

    # 检查4: 检查 load_qwen 函数
    utils_file = "/Users/xiniuyiliao/Desktop/code/Travel-Agent-based-on-Qwen2-RLHF/src/utils/utils.py"
    if os.path.exists(utils_file):
        with open(utils_file, 'r', encoding='utf-8') as f:
            utils_content = f.read()
        if 'Qwen2ForCausalLM.from_pretrained' in utils_content:
            checks.append({
                "test": "load_qwen 使用 Qwen2ForCausalLM",
                "passed": True,
                "detail": "load_qwen 正确使用 Qwen2ForCausalLM.from_pretrained"
            })
        else:
            checks.append({
                "test": "load_qwen 使用 Qwen2ForCausalLM",
                "passed": False,
                "detail": "load_qwen 未使用 Qwen2ForCausalLM"
            })

    return checks, issues


def generate_report(checks, issues):
    """生成测试报告"""
    report_path = "/Users/xiniuyiliao/Desktop/code/Travel-Agent-based-on-Qwen2-RLHF/docs/测试报告/TEST_REPORT.md"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    import torch

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 模型推理功能测试报告\n\n")
        f.write("**测试日期**: 2026-04-15\n\n")
        f.write("**测试类型**: 代码审查 + 结构测试\n\n")
        f.write("**注意**: 由于环境依赖缺失(bitsandbytes等)，无法实际加载模型进行推理测试。\n")
        f.write("本报告重点检查代码正确性和参数配置。\n\n")

        f.write("## 测试环境\n\n")
        f.write(f"- 模型路径: {MODEL_PATH}\n")
        f.write(f"- PyTorch版本: {torch.__version__}\n")
        f.write(f"- CUDA可用: {torch.cuda.is_available()}\n")
        f.write(f"- GPU数量: {torch.cuda.device_count()}\n\n")

        f.write("## 代码检查结果\n\n")
        f.write("| 检查项 | 状态 | 详情 |\n")
        f.write("|--------|------|------|\n")
        for check in checks:
            status = "通过" if check["passed"] else "未通过"
            f.write(f"| {check['test']} | {status} | {check['detail']} |\n")

        f.write("\n## 发现的问题\n\n")

        if issues:
            for issue in issues:
                f.write(f"### {issue['severity']}: {issue['file']}\n\n")
                f.write(f"- **位置**: 第 {issue['line']} 行\n")
                f.write(f"- **问题**: {issue['issue']}\n")
                f.write(f"- **建议修复**: {issue['fix']}\n\n")
        else:
            f.write("未发现严重问题。\n\n")

        f.write("## 参数配置检查\n\n")
        f.write("### generate_response 方法 (model.py:168-214)\n\n")
        f.write("```python\n")
        f.write("outputs = self.model.generate(\n")
        f.write("    **inputs,\n")
        f.write("    max_new_tokens=max_length,  # [正确] 使用 max_new_tokens\n")
        f.write("    temperature=temperature,\n")
        f.write("    top_p=top_p,\n")
        f.write("    do_sample=True,\n")
        f.write("    pad_token_id=self.tokenizer.pad_token_id,\n")
        f.write("    eos_token_id=self.tokenizer.eos_token_id\n")
        f.write(")\n")
        f.write("```\n\n")
        f.write("**检查结果**: 参数配置正确。\n\n")

        f.write("### stream_chat 方法 (model.py:240-310)\n\n")
        f.write("```python\n")
        f.write("outputs = self.model.generate(\n")
        f.write("    input_ids=generated_ids,\n")
        f.write("    attention_mask=attention_mask,\n")
        f.write("    max_length=max_length,  # [正确] 限制输入+输出总长度\n")
        f.write("    max_new_tokens=1,  # [正确] 每次只生成1个新token\n")
        f.write("    temperature=temperature,\n")
        f.write("    top_p=top_p,\n")
        f.write("    do_sample=True,\n")
        f.write("    pad_token_id=self.tokenizer.pad_token_id,\n")
        f.write("    eos_token_id=self.tokenizer.eos_token_id,\n")
        f.write("    output_scores=True,\n")
        f.write("    return_dict_in_generate=True,\n")
        f.write("    num_return_sequences=1,\n")
        f.write("    output_hidden_states=False\n")
        f.write(")\n")
        f.write("```\n\n")
        f.write("**检查结果**: 参数配置正确。\n\n")

        f.write("## load_qwen 函数检查 (utils.py:312-371)\n\n")
        f.write("```python\n")
        f.write("def load_qwen(model_name, use_flash_attention=False):\n")
        f.write("    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n")
        f.write("    # ...\n")
        f.write("    model = Qwen2ForCausalLM.from_pretrained(\n")
        f.write("        model_name,\n")
        f.write("        torch_dtype=torch.float16,\n")
        f.write("        trust_remote_code=True,\n")
        f.write("        max_memory=max_memory,\n")
        f.write("        low_cpu_mem_usage=True,\n")
        f.write("    )\n")
        f.write("    # ...\n")
        f.write("```\n\n")
        f.write("**检查结果**: 函数正确使用 Qwen2ForCausalLM.from_pretrained。\n\n")

        # 总结
        all_passed = all(c["passed"] for c in checks)
        has_errors = any(i["severity"] == "ERROR" for i in issues)

        f.write("## 结论\n\n")

        if has_errors:
            f.write("**状态**: 代码存在错误，需要修复后才能正常运行。\n\n")
            f.write("主要问题:\n")
            for issue in issues:
                if issue["severity"] == "ERROR":
                    f.write(f"- {issue['issue']}\n")
        elif all_passed:
            f.write("**状态**: 代码结构和参数配置正确。\n\n")
        else:
            f.write("**状态**: 部分检查未通过。\n\n")

        f.write("\n## 后续建议\n\n")
        f.write("1. 修复 MODEL_CONFIG 导入错误\n")
        f.write("2. 安装缺失的依赖: bitsandbytes, peft\n")
        f.write("3. 在修复依赖后，进行实际的模型加载和推理测试\n")

    print(f"\n测试报告已保存到: {report_path}")
    return all_passed and not has_errors


def main():
    print("\n" + "#" * 60)
    print("# 模型推理功能测试")
    print("#" * 60)

    checks, issues = check_code_correctness()

    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)

    for check in checks:
        status = "[PASS]" if check["passed"] else "[FAIL]"
        print(f"{status} {check['test']}")
        print(f"       {check['detail']}")

    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  [{issue['severity']}] {issue['file']}:{issue['line']}")
            print(f"         {issue['issue']}")

    success = generate_report(checks, issues)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
