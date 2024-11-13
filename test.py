import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint, NonIncrementalGrammarConstraint
#from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from logits import GrammarConstrainedLogitsProcessor, NegativeConstraintNGramLogitsProcessor
from grammar import IncrementalTokenRecognizer as IncrementalGrammarConstraint

import logging

#logger = logging.getLogger('test')
logging.basicConfig(
    level=logging.DEBUG,
    filename="test.log",
    encoding="utf-8",
    filemode="a",
    )

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "deepseek-ai/deepseek-coder-1.3b-base"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(
        device
    )  # Load model to defined device
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Load grammar
    with open("json.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar) #NegativeConstraintNGramLogitsProcessor(tokenizer(["\n"]).input_ids) #

    # Generate
    prefix1 = "This is a valid json string for http request:"
    prefix2 = "This is a valid json string for shopping cart:"
    input_ids = tokenizer(
        [prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=60,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)

    """
    'This is a valid json string for http request:{ "request": { "method": "GET", "headers": [], "content": "Content","type": "application" }}
    'This is a valid json string for shopping cart:This is a valid json string for shopping cart:{ "name": "MyCart", "price": 0, "value": 1 }
    """