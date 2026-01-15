import torch
import re
from collections import defaultdict
import os
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

from .fact_extractor_tool import DocListFactExtractorTool, FactExtractorConfig

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False

    # retrieval
    search_url: str = None
    topk: int = 3

    # fact extractor (tool-side, masked by <documents> ... </documents>)
    facts_enabled: bool = True
    fact_extractor_mode: str = "heuristic"  # "heuristic" | "llm"
    fact_llm_url: str = None               # required when mode == "llm"
    fact_concurrency: int = 8
    max_fact_new_tokens: int = 256
    fact_timeout_s: float = 30.0
    fact_temperature: float = 0.0
    fact_top_p: float = 0.95
    fact_top_k: int = 0

    # budgets to prevent context blow-up
    docs_max_chars: int = 12000
    max_body_chars: int = 4000
    facts_max_chars: int = 1200


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        # Tool-side fact extractor (outside policy optimization)
        self.fact_extractor = None
        if getattr(config, 'facts_enabled', False) and getattr(config, 'fact_extractor_mode', 'heuristic') == 'llm':
            self.fact_extractor = DocListFactExtractorTool(
                tokenizer=tokenizer,
                cfg=FactExtractorConfig(
                    fact_llm_url=getattr(config, 'fact_llm_url', None),
                    timeout_s=float(getattr(config, 'fact_timeout_s', 30.0)),
                    concurrency=int(getattr(config, 'fact_concurrency', 8)),
                    temperature=float(getattr(config, 'fact_temperature', 0.0)),
                    top_p=float(getattr(config, 'fact_top_p', 0.95)),
                    top_k=int(getattr(config, 'fact_top_k', 0)),
                    max_fact_new_tokens=int(getattr(config, 'max_fact_new_tokens', 256)),
                    max_body_chars=int(getattr(config, 'max_body_chars', 4000)),
                    facts_max_chars=int(getattr(config, 'facts_max_chars', 1200)),
                ),
            )

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _now(self) -> float:
        """High-resolution wall clock for profiling."""
        return time.perf_counter()

    def _acc_time(self, key: str, dt_s: float) -> None:
        """Accumulate timing into self.timing_raw if available."""
        timing_raw = getattr(self, "timing_raw", None)
        if isinstance(timing_raw, dict):
            timing_raw[key] = float(timing_raw.get(key, 0.0)) + float(dt_s)


    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            t0 = self._now()
            t0 = self._now()
            gen_output = self._generate_with_gpu_padding(rollings_active)
            self._acc_time("agent_llm_s", self._now() - t0)
            self._acc_time("agent_llm_s", self._now() - t0)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        meta_info["active_traj_num_list"] = active_num_list
        if os.environ.get("VERL_DEBUG_ACTIVE_TRAJ_NUM", "0") == "1":
            print("ACTIVE_TRAJ_NUM:", active_num_list)

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM

        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding
            active_mask: list[0/1] marking which envs are active
            do_search: whether to call retrieval service

        Returns:
            next_obs, dones, valid_action, is_search
        """
        cur_actions, contents = self.postprocess_predictions(predictions)

        # Be robust: if caller doesn't provide active_mask, treat all as active.
        if active_mask is None:
            active_mask = [1] * len(cur_actions)

        next_obs, dones, valid_action, is_search = [], [], [], []

        # Only issue tool calls for ACTIVE envs that choose SEARCH.
        search_idxs = [i for i, (a, actv) in enumerate(zip(cur_actions, active_mask)) if actv and a == "search"]
        search_queries = [contents[i] for i in search_idxs]

        if do_search and search_queries:
            t0 = self._now()
            search_results = self.batch_search(search_queries)
            self._acc_time("agent_search_s", self._now() - t0)
            assert len(search_results) == len(search_queries)
        else:
            # Keep alignment with search_queries length.
            search_results = [""] * len(search_queries)

        # Facts must ALWAYS be defined (fixes UnboundLocalError).
        # This will call tool-LLM when configured, otherwise heuristic fallback.
        facts_results = []
        if search_results:
            t0 = self._now()
            facts_results = self._extract_facts_batch(search_results)
            self._acc_time("agent_fact_s", self._now() - t0)

        # Use an index pointer (NOT pop) to avoid misalignment when some envs are inactive.
        search_ptr = 0

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue

            if action == "answer":
                next_obs.append("")
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
                continue

            if action == "search":
                j = search_ptr
                search_ptr += 1

                docs_raw = (search_results[j] if j < len(search_results) else "").strip()
                docs_raw = self._truncate(docs_raw, getattr(self.config, "docs_max_chars", 12000))

                facts_raw = (facts_results[j] if j < len(facts_results) else "").strip()
                facts_raw = self._truncate(facts_raw, getattr(self.config, "facts_max_chars", 1200))

                # Embed <facts> INSIDE <documents> to keep state_masking behavior consistent.
                if facts_raw:
                    docs_payload = f"{docs_raw}\n\n<facts>\n{facts_raw}\n</facts>"
                else:
                    docs_payload = docs_raw

                next_obs.append(
                    f"\n\n<documents>{docs_payload}</documents>\n\n"
                    "Update <memory> based on the evidence above, then decide next step with "
                    "<search>...</search> or <answer>...</answer>.\n"
                )
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
                continue

            # Invalid action format
            next_obs.append(
                "\nMy previous action is invalid. "
                "If I want to search, I should put the query between <search> and </search>. "
                "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                "Let me try again.\n"
            )
            dones.append(0)
            valid_action.append(0)
            is_search.append(0)

        # Sanity: we should have consumed exactly the number of active searches.
        assert search_ptr == len(search_queries)

        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                matches = re.findall(pattern, prediction, flags=re.DOTALL)
                if matches:
                    action = matches[-1][0]
                    content = matches[-1][1].strip()
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def _truncate(self, text: str, max_chars: int) -> str:
        if not text or max_chars is None:
            return text or ""
        return text[:max_chars].rstrip()

    def _extract_facts_heuristic(self, docs_text: str, facts_per_doc: int = 3) -> str:
        """Fallback heuristic facts: take first N sentences from each doc body."""
        if not docs_text:
            return ""

        # split docs by header lines
        chunks = re.split(r"\n(?=Doc\s+\d+\s+\(Title:)", "\n" + docs_text.strip())
        facts_lines = []
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            lines = ch.splitlines()
            header = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            # naive sentence split
            sents = re.split(r"(?<=[\.\!\?])\s+", body)
            picked = [s.strip() for s in sents if s.strip()][:facts_per_doc]
            if picked:
                facts_lines.append(header)
                for s in picked:
                    facts_lines.append(f"- {s}")
        out = "\n".join(facts_lines).strip()
        return self._truncate(out, getattr(self.config, "facts_max_chars", 1200))

    def _extract_facts_batch(self, docs_text_list: List[str]) -> List[str]:
        """Extract facts for each docs_text in the list; tool-call if enabled, else heuristic."""
        if not getattr(self.config, "facts_enabled", False):
            return [""] * len(docs_text_list)

        mode = getattr(self.config, "fact_extractor_mode", "heuristic")
        if mode == "llm" and self.fact_extractor is not None and getattr(self.config, "fact_llm_url", None):
            try:
                return self.fact_extractor.extract_facts_batch(docs_text_list)
            except Exception:
                # fail-safe fallback
                return [self._extract_facts_heuristic(t) for t in docs_text_list]

        # heuristic default
        return [self._extract_facts_heuristic(t) for t in docs_text_list]

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()
    
    @staticmethod
    def _truncate_text(text: str, max_length: int = 512) -> str:
        return text[:max_length] if text else text

    @staticmethod
    def _generate_summary(text: str, max_length: int = 512) -> str:
        if not text:
            return ""
        truncated = LLMGenerationManager._truncate_text(text, max_length)
        sentences = re.split(r'(?<=[.!?。！？])\s+', truncated)
        summary = ' '.join([s.strip() for s in sentences if s.strip()][:2])
        return summary if summary else truncated

    def _passages2string(self, retrieval_result):
        """Format retrieved passages into MemSearcher-friendly doc blocks.

        Output format (per doc):
            Doc k (Title: xxx)
            body...

        This exact header style is required by the LLM fact extractor.
        """
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            document = doc_item.get('document', doc_item) if isinstance(doc_item, dict) else {}
            contents = document.get('contents') or document.get('text', '')
            title = document.get('title')
            if not title and contents:
                title = contents.split("\n")[0]
            summary_or_text = document.get('summary')
            if not summary_or_text:
                summary_or_text = self._generate_summary(contents)
            summary_or_text = self._truncate_text(summary_or_text, 512)
            title = title if title else "N/A"
            format_reference += f"Doc {idx+1}(Title: {title}) {summary_or_text}\n"

        return format_reference.strip()

