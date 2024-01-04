import sys
import json

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"

def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            d = json.loads(line)
            input_data.append(d)
    return input_data


def main(
    load_8bit: bool = False,
    base_model: str = "",
    # the infer data, if not exists, infer the default instructions in code
    instruct_dir: str = "",
    use_lora: bool = True,
    lora_weights: str = "tloen/alpaca-lora-7b",
    # The prompt template to use, will default to med_template.
    prompt_template: str = "med_template",
):
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        print(f"using lora {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    def infer_from_json(instruct_dir):
        input_data = json.load(open(instruct_dir, 'r', encoding='utf-8'))
        # input_data = load_instruction(instruct_dir)
        result = []
        for i,d in enumerate(input_data):
            instruction = d["instruction"]
            gold = d["output"]
            model_output = evaluate(instruction)
            temp = {
                'instruction':instruction,
                'reference':gold,
                'output':model_output
            }
            result.append(temp)
            print(f"# inferring [{i+1}/{len(input_data)}]")

        if 'en' in instruct_dir:
            outpath = './output/en_output.json'
        else:
            outpath = './output/zh_output.json'
        with open(outpath,'w',encoding='utf-8') as f:
            json.dump(result,f,indent=3,ensure_ascii=False)
        print('\n',outpath,' down!!!')

    if instruct_dir != "":
        infer_from_json(instruct_dir)
    else:
        for instruction in [
            "请根据以下描述来回答医学问题或给出建议：我感冒了，怎么治疗",
            "请根据以下描述来回答医学问题或给出建议：一个患有肝衰竭综合征的病人，除了常见的临床表现外，还有哪些特殊的体征？",
            "请根据以下描述来回答医学问题或给出建议：急性阑尾炎和缺血性心脏病的多发群体有何不同？",
            "请根据以下描述来回答医学问题或给出建议：小李最近出现了心动过速的症状，伴有轻度胸痛。体检发现P-R间期延长，伴有T波低平和ST段异常",
        ]:
            print("Instruction:", instruction)
            print("Response:", evaluate(instruction))
            print()


if __name__ == "__main__":
    fire.Fire(main)
