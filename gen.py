import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint, NonIncrementalGrammarConstraint
#from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from logits import GrammarQueryConstrainedLogitsProcessor, NegativeConstraintNGramLogitsProcessor
from grammar import IncrementalTokenRecognizer as IncrementalGrammarConstraint

import logging

import weggli

def generate(inputs, model_id, grammar_file, weggli_query, **kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(
        device
    )  # Load model to defined device
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Load grammar
    with open(grammar_file, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_query_processor = GrammarQueryConstrainedLogitsProcessor(grammar, weggli.weggli_query_constraint(weggli_query)) 

    # Generate

    input_ids = tokenizer(
        inputs, add_special_tokens=True, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=120,
        logits_processor=[grammar_query_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generations

def main():

    inputs = ["""
              
Complete the following code using `memcpy`.

```
#include <stdio.h>
#include <string.h>
              
int main() {

    char str1[] = "Hello ";
    char str2[] = "there!";

    puts("str1 before memcpy ");
    puts(str1);

    // Copies contents of str2 to str1
"""]
    
    query = """
    {
        memcpy(_,_,6);
    }
"""

    print(generate(inputs, "deepseek-ai/deepseek-coder-1.3b-base", "c.ebnf", query))
    

if __name__ == "__main__":
    main()

