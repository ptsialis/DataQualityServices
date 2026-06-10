#Copyright 2024 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_model(model = "", quantization='4bit'):
    """
    Load a causal language model + tokenizer from Hugging Face and wrap it in a
    text-generation pipeline, optionally using bitsandbytes quantization.

    Args:
        model (str): Hugging Face model id or local path (e.g., "microsoft/Phi-4").
        quantization (str): One of {"8bit", "4bit", "16bit", ""}.
            - "8bit": load_in_8bit via BitsAndBytesConfig
            - "4bit": load_in_4bit via BitsAndBytesConfig (compute in bfloat16)
            - "16bit": placeholder path here (uses BitsAndBytesConfig as written)
            - "": no quantization (full-precision weights)

    Returns:
        transformers.Pipeline: A text-generation pipeline using the loaded model/tokenizer.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    except ImportError as exc:
        raise ImportError(
            "Failed to import transformers. Ensure compatible versions of transformers and huggingface_hub are installed."
        ) from exc

    # Load tokenizer first; ensure a pad token id exists to avoid generation warnings.
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # disable terminal warning about missing pad token

    # Build a bitsandbytes quantization config depending on the requested mode.
    # (Requires bitsandbytes + compatible GPU for 8/4-bit.)
    if quantization == '8bit':
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantization == '4bit':
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16  # compute in bfloat16 for speed/accuracy tradeoff
        )
    elif quantization == '16bit':
        # Note: This path assumes BitsAndBytesConfig supports 16-bit flag as used below.
        # Some setups prefer torch_dtype=torch.float16 on from_pretrained instead.
        quant_config = BitsAndBytesConfig(load_in_16bit=True)

    # Load model weights. If a quantization mode was chosen, pass the quant config;
    # otherwise load the model without quantization. device_map="auto" spreads layers across available devices.
    def _load_model(device_map):
        if not quantization == "":
            return AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=quant_config,
                device_map=device_map,
            )
        return AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
        )

    device_map = 'auto' if torch.cuda.is_available() else 'cpu'
    try:
        llm_model = _load_model(device_map)
    except RuntimeError as exc:
        if torch.cuda.is_available() and 'out of memory' in str(exc).lower():
            logger.warning("CUDA out of memory while loading model; retrying on CPU.")
            torch.cuda.empty_cache()
            llm_model = _load_model('cpu')
        else:
            raise

    # Wrap into a text-generation pipeline for easy .__call__(...) usage.
    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
    )
    return pipe


# def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", quantization=""):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token_id = tokenizer.eos_token_id  # disable terminal warning

#     model_kwargs = {"device_map": "auto"}  # always applied

#     if quantization == "8bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_8bit=True
#         )
#     elif quantization == "4bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
#     elif quantization == "16bit":
#         model_kwargs["torch_dtype"] = torch.float16
#     elif quantization != "":
#         raise ValueError(f"Unsupported quantization option: {quantization}")

#     model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )
#     return pipe

# def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", quantization=""):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token_id = tokenizer.eos_token_id  # disable terminal warning

#     model_kwargs = {"device_map": "auto"}  # always applied

#     if quantization == "8bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_8bit=True
#         )
#     elif quantization == "4bit":
#         model_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
#     elif quantization == "16bit":
#         model_kwargs["torch_dtype"] = torch.float16
#     elif quantization != "":
#         raise ValueError(f"Unsupported quantization option: {quantization}")

#     model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )
#     return pipe





def ask_model_mes(messages, pipe=None, max_new_tokens=10000):
    """
    Call a chat-style text-generation pipeline with an already-built list of messages.

    Args:
        messages (list[dict]): Chat messages in the format expected by your pipeline, e.g.:
            [
              {"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."},
              ...
            ]
        pipe: A Hugging Face `pipeline("text-generation", ...)` (or compatible) object
              that supports chat-format input.
        max_new_tokens (int): Maximum number of tokens the model may generate.

    Returns:
        str: The assistant's textual content from the last generated chat turn.

    Notes:
        - This assumes the pipeline returns a list with one item, whose structure is:
          outputs[0]["generated_text"]  -> a list of chat turns (dicts),
          and the final generated turn is at index -1 with key 'content'.
    """
    # Guard: model pipeline must be provided
    if pipe is None:
        return "Model was not loaded! Use load_model() before running this function."

    # Invoke the model with chat messages; pass generation length
    outputs = pipe(messages, max_new_tokens=max_new_tokens)

    # Extract the last generated message's content from the pipeline output
    response = outputs[0]["generated_text"][-1]['content']
    return response


def ask_model_prompt(prompt, system_prompt=None, pipe=None):
    """
    Build a chat message list from a raw user prompt (and optional system prompt),
    then call a chat-style text-generation pipeline.

    Args:
        prompt (str): The user instruction/question.
        system_prompt (str | None): Optional system message to steer behavior.
        pipe: A Hugging Face `pipeline("text-generation", ...)` (or compatible) object.

    Returns:
        str: The assistant's textual content from the last generated chat turn.
             Returns empty string if generation completely fails.

    Generation params:
        - max_new_tokens=512 (reduced for stability)
        - temperature=0.7
        - top_p=0.9
        - top_k=50
        - do_sample=True
        - repetition_penalty=1.1
    """
    # Guard: model pipeline must be provided
    if pipe is None:
        return "Model was not loaded! Use load_model() before running this function."
    else:
        # Construct chat messages depending on whether a system prompt is provided
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        # Call the model with stable generation parameters
        try:
            outputs = pipe(
                messages,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1
            )
        except RuntimeError as e:
            if "probability tensor" in str(e) or "inf" in str(e) or "nan" in str(e):
                logger.warning(f"Model generation failed with probability tensor error: {e}. Retrying with deterministic generation...")
                try:
                    outputs = pipe(
                        messages,
                        max_new_tokens=128,
                        temperature=1.0,
                        do_sample=False
                    )
                except Exception as retry_error:
                    logger.error(f"Deterministic generation also failed: {retry_error}")
                    # Return empty string instead of crashing
                    return ""
            else:
                logger.error(f"Model generation failed: {e}")
                return ""
        except Exception as e:
            logger.error(f"Unexpected error during model generation: {e}")
            return ""

        try:
            # Extract the assistant's final generated message content
            response = outputs[0]["generated_text"][-1]['content']
            return response if response else ""
        except (IndexError, KeyError, TypeError) as e:
            logger.error(f"Failed to extract response from model output: {e}")
            return ""
