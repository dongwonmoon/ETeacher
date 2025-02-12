import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import time
from config.prompts import teacher_prompt

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    framework="pt",
    do_sample=False,
    batch_size=1,
)


def teach(user_prompt: str, caption: str, max_new_tokens: int = 20) -> str:
    org_sys = teacher_prompt[0]["content"]
    org_usr = teacher_prompt[1]["content"]

    teacher_prompt[0]["content"] = teacher_prompt[0]["content"].format(caption)
    teacher_prompt[1]["content"] = user_prompt

    with torch.inference_mode():
        output = pipe(teacher_prompt, max_new_tokens=max_new_tokens)[0][
            "generated_text"
        ][-1]["content"]

    teacher_prompt[0]["content"] = org_sys
    teacher_prompt[1]["content"] = org_usr

    return output
