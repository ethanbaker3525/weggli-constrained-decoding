import copy
import math
import os
import pprint

import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings

import weggli

logger = logging.getLogger(__name__)


class GrammarQueryConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint, query_constraint, valid_token_start_idx=None, device=None):
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_parsing_states = None
        self.valid_token_start_idx = valid_token_start_idx
        self.device = device
        self.query_constraint = query_constraint

    def mask_logits(self, logits, device):
        masked_logits = logits.clone()
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
       #print(masked_logits)
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_parsing_states, device
        )

        # if the logits size of the model is more than the tokennizer vocab
        # we artificially expand the acceptance tensor and block everything
        # beyond the tokenizer vocab size
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = masked_logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        # acceptance is a tensor of shape (batch_size, vocab_size)
        # get the indices of the accepted tokens
        # do the following operation only in debug mode
        if False:
            # convert acceptance to numpy array
            batch_size, vocab_size = acceptance.shape
            acceptance_np = acceptance.cpu().numpy()
            accepted_x, accepted_y = acceptance_np.nonzero()
            # dict of {batch_index: [accepted_token_indices]}
            # initialize the dict with empty list
            accepted_token_indices = {i: [] for i in range(batch_size)}
            for x, y in zip(accepted_x, accepted_y):
                accepted_token_indices[x].append(y)
            logger.debug("Accepted token indices for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_token_indices))
            # convert token_ids to tokens
            accepted_tokens = {
                i: [
                    self.grammar_constraint.tokenizer.decode([token_id])
                    for token_id in token_ids
                ]
                for i, token_ids in accepted_token_indices.items()
            }
            logger.debug("Accepted tokens for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_tokens))
        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
        return masked_logits

    def process_logits(self, input_ids, scores):
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_parsing_states is None:
            self.batch_parsing_states = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_parsing_state()
                )
                for _ in range(len(input_ids))
            ]

        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)

        logger.debug("input_ids: \n" + pprint.pformat(input_ids))
        # logger.debug("scores: \n" + pprint.pformat(scores))
        logger.debug("last_size: \n" + pprint.pformat(self.last_size))
        logger.debug(
            "num of stacks: \n"
            + pprint.pformat(
                [len(acc_state.stacks) for acc_state in self.batch_parsing_states]
            )
        )
        # logger.debug("stacks: \n" + pprint.pformat(self.batch_parsing_states.stacks))


        self.batch_parsing_states = (
            self.grammar_constraint.update_state_with_batch_token_seqs(
                input_ids, self.batch_parsing_states, self.valid_token_start_idx
            )
        )

        logger.debug(f"input_ids: {input_ids}")
        #print(self.grammar_constraint.tokenizer.batch_decode(input_ids, skip_special_tokens=True))

        #print(input_ids)
        
        #print([s.partial_utf8 for s in self.batch_parsing_states])

        top_k_scores, top_k_ids = torch.topk(scores, 10, dim=-1)
        #print(top_k_ids)
        #quit()
        #print(self.grammar_constraint.tokenizer.batch_decode(scores, skip_special_tokens=True))

        #print("___")

        masked_scores = self.mask_logits(scores, device)

        
        #print(top_k_ids)
        #quit()
        

        masked_scores = check_negative_constraint_str(input_ids, masked_scores, self.grammar_constraint.tokenizer, self.query_constraint)

        return masked_scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)

def get_batch_possible_completion_tensors(input_ids, masked_scores, top_k=5):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for batch in range(len(masked_scores)):

        input_ids_batch = input_ids[batch].to(device).long()  # Assuming input_ids are integers (torch.long)
        masked_scores_batch = masked_scores[batch].to(device)  # Assuming scores are on GPU

        # Generate indices and create mask, replacing -inf with -1
        indices = torch.arange(len(masked_scores_batch), device=device).long()  # Ensure indices are integers
        mask = torch.where(masked_scores_batch != float("-inf"), indices, torch.tensor(-1, device=device, dtype=torch.int64))

        # Get top-k indices based on masked_scores_batch
        top_k_scores, top_k_indices = torch.topk(masked_scores_batch, top_k, dim=0, largest=True)

        # Initialize the result tensor with -1 (or another placeholder value)
        result = torch.full((len(masked_scores_batch), input_ids_batch.size(0) + 1), -1, device=device, dtype=torch.int64)

        # Mask for top-k indices (we only want to keep sequences corresponding to the top-k scores)
        valid_mask = torch.isin(mask, top_k_indices)  # This ensures only top-k indices are processed

        # Get valid indices (where mask is not -1 and in the top-k)
        valid_indices = mask[valid_mask]

        # Now, create the concatenated sequences only for valid top-k indices
        concatenated_sequences = torch.cat([input_ids_batch.unsqueeze(0).repeat(valid_indices.size(0), 1), valid_indices.unsqueeze(1)], dim=1)

        # Update the result tensor with concatenated sequences at the valid positions
        result[valid_mask] = concatenated_sequences

        # Append the result to the results list
        results.append((result, top_k_indices))  # Also keep track of the top-k indices

    return results


# Function to decode only valid sequences (top-k) from the batch
def check_negative_constraint_str(input_ids, masked_scores, tokenizer, query_constraint, top_k=5):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoded_results = []

    for batch, top_k_indices in get_batch_possible_completion_tensors(input_ids, masked_scores, top_k):

        # Find padding token ID for the tokenizer
        padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Replace -1 (invalid tokens) with the padding token ID
        result = torch.where(batch == -1, padding_token_id, batch)

        # Move to CPU for decoding
        result_cpu = result.cpu()

        valid_sequences = result_cpu[result_cpu != padding_token_id]

        # Reshape valid sequences dynamically based on the number of valid elements
        # This ensures we don't encounter shape mismatches
        if valid_sequences.size(0) > 0:
            # Reshape valid sequences to group them correctly for batch decoding
            # We reshape it as needed for the batch decode
            # print(valid_sequences)
            #print(valid_sequences.shape)
            #print(result_cpu.shape)
            if device == "cuda":
                valid_sequences = valid_sequences.view(-1, result_cpu.size(1))
            

            # Decode valid sequences only
            #print(valid_sequences)
            decoded_batch = tokenizer.batch_decode(valid_sequences.tolist(), skip_special_tokens=True)
            decoded_results.extend(decoded_batch)
        # At this point, you can now use `top_k_indices` to mask the original `masked_scores` based on the decoded results
        # Example: you can update masked_scores based on some condition related to the decoded strings
        # (For example, applying a new score mask based on certain keywords or conditions in the decoded strings)
        for idx, top_k_index in enumerate(top_k_indices):
            # Here, top_k_index corresponds to the original index of the top-k element in the masked_scores
            # You can mask scores or apply adjustments here using `top_k_index`
            #print(input_ids)
            #print(masked_scores)
            #print(top_k_index)
            #print(idx)
            #print(decoded_results)
            output = tokenizer.decode(input_ids.tolist()[0] + [top_k_index.tolist()])
            if query_constraint(output):
                masked_scores[top_k_index] = float("-inf")



    print(decoded_results)
    quit()
    return masked_scores  # Return both decoded results and modified masked_scores (if needed)



class NegativeConstraintNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, negative_constraints):
        self.negative_constraints = dict()
        for constraint in negative_constraints:
            self.negative_constraints[tuple(constraint[:-1])] = constraint[-1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        for i in range(num_batch_hypotheses):
            for ngram in self.negative_constraints:
                # print(ngram)
                ngram_size = len(ngram) + 1
                if cur_len + 1 < ngram_size:
                    continue
                prev_ngram_tuple = tuple(input_ids[i, cur_len + 1 - ngram_size:cur_len].tolist())
                banned_token = self.negative_constraints.get(prev_ngram_tuple, None)
                # print(banned_token)
                if banned_token is not None:
                    scores[i, banned_token] = float("-inf")

        return scores