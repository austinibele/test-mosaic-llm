from typing import Any, Dict, Tuple
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


class InstructionTextGenerationPipeline:
    def __init__(
        self,
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=None,
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
        if tokenizer.pad_token_id is None:
            warnings.warn(
                "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
            )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)

        self.generate_kwargs = {
            "temperature": 0.5,
            "top_p": 0.92,
            "top_k": 0,
            "max_new_tokens": 512,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,  # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
        }

    def format_instruction(self, instruction):
        return PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)

    def __call__(
        self, instruction: str, **generate_kwargs: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        s = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
        input_ids = self.tokenizer(s, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        gkw = {**self.generate_kwargs, **generate_kwargs}
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gkw)
        # Slice the output_ids tensor to get only new tokens
        new_tokens = output_ids[0, len(input_ids[0]) :]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text
