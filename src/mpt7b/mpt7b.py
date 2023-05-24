import torch
from threading import Thread
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from .instruction_text_generation_pipeline import InstructionTextGenerationPipeline

generate = InstructionTextGenerationPipeline(
    "mosaicml/mpt-7b-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
stop_token_ids = generate.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class Mpt7b:
    def infer(self, instruction, temperature=0.3, top_p=0.95, top_k=0, max_new_tokens=2000):
        # Tokenize the input
        input_ids = generate.tokenizer(
            generate.format_instruction(instruction), return_tensors="pt"
        ).input_ids
        input_ids = input_ids.to(generate.model.device)

        # Initialize the streamer and stopping criteria
        streamer = TextIteratorStreamer(
            generate.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        stop = StopOnTokens()

        if temperature < 0.1:
            temperature = 0.0
            do_sample = False
        else:
            do_sample = True

        gkw = {
            **generate.generate_kwargs,
            **{
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
                "streamer": streamer,
                "stopping_criteria": StoppingCriteriaList([stop]),
            },
        }

        response = ''
        

        def generate_and_signal_complete():
            generate.model.generate(**gkw)

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        for new_text in streamer:
            response += new_text
    
        return response