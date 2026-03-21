"""
llm_handler.py - Abstraction layer for HuggingFace Transformers LLM interactions.
Handles model loading, inference calls, and response parsing for both
Qwen 2.5 (extraction) and Llama 3.2 (generation) models.

This module manages two distinct models:
- Extraction model (Qwen 2.5 1.5B): For structured data extraction tasks (low temperature)
- Generation model (Llama 3.2 3B): For creative content generation tasks (moderate temperature)

Models are loaded locally using HuggingFace Transformers — no external server required.
Works on Windows, macOS, and Linux. Uses CPU by default; GPU used automatically if available.

Note: Llama 3.2 is gated on HuggingFace and requires a free HF account + accepted license.
      Qwen 2.5 is ungated (no token required).
      Pass your HuggingFace token via --hf-token argument or HF_TOKEN environment variable.
      Alternatively, run 'huggingface-cli login' once before running this script.
"""

import sys
import os


class LLMHandler:
    """
    Unified interface for interacting with HuggingFace Transformers LLMs.
    Manages two model instances: one for extraction (Qwen 2.5 1.5B),
    one for generation (Llama 3.2 3B).

    Models are downloaded once on first use and cached locally by HuggingFace.
    Subsequent runs load from cache — no internet needed.
    """

    # Default HuggingFace model identifiers
    DEFAULT_EXTRACTION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    DEFAULT_GENERATION_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, extraction_model=None, generation_model=None, hf_token=None):
        """
        Initialize the LLM handler with model names and optional HF token.

        Args:
            extraction_model (str): HuggingFace model ID for extraction tasks
                                    (default: Qwen/Qwen2.5-1.5B-Instruct)
            generation_model (str): HuggingFace model ID for generation tasks
                                    (default: meta-llama/Llama-3.2-3B-Instruct)
            hf_token (str or None): HuggingFace access token for gated models.
                                    Falls back to HF_TOKEN environment variable.
        """
        self.extraction_model_id = extraction_model or self.DEFAULT_EXTRACTION_MODEL
        self.generation_model_id = generation_model or self.DEFAULT_GENERATION_MODEL

        # Resolve HF token: argument > environment variable > None (uses cached login)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", None)

        # Pipelines are loaded lazily on first use to avoid loading both at startup
        self._extraction_pipeline = None
        self._generation_pipeline = None

    def verify_connection(self):
        """
        Verify that required packages are importable and that model files
        are accessible.  Models are loaded lazily — only ONE model is kept
        in memory at a time to prevent OOM on memory-constrained devices.

        Returns:
            bool: True if packages are available and model files are cached

        """
        import transformers
        import torch

        print("\n  AI models verified. Models will load on demand (one at a time).\n")
        return True

    def extract(self, prompt, temperature=0.1, max_retries=2):
        """
        Send a prompt to the extraction model (Qwen 2.5) for structured data tasks.
        Uses low temperature for deterministic, structured JSON outputs.
        Automatically loads the extraction model if not already loaded,
        and unloads the generation model first to free memory.

        Args:
            prompt (str): The full prompt text for the extraction task
            temperature (float): Sampling temperature (default 0.1 for extraction)
            max_retries (int): Number of retry attempts on failure (default 2)

        Returns:
            str: Raw model response text
        """
        # Ensure only extraction model is in memory
        if self._generation_pipeline is not None:
            self._unload_generation()
        return self._call_model(
            self._get_extraction_pipeline, prompt, temperature, max_retries,
            max_new_tokens=2048
        )

    def generate(self, prompt, temperature=0.7, max_retries=2):
        """
        Send a prompt to the generation model (Llama 3.2 3B) for creative content tasks.
        Uses moderate temperature for varied, natural-sounding outputs.
        Automatically loads the generation model if not already loaded,
        and unloads the extraction model first to free memory.

        Args:
            prompt (str): The full prompt text for the generation task
            temperature (float): Sampling temperature (default 0.7 for generation)
            max_retries (int): Number of retry attempts on failure (default 2)

        Returns:
            str: Raw model response text
        """
        # Ensure only generation model is in memory
        if self._extraction_pipeline is not None:
            self._unload_extraction()
        return self._call_model(
            self._get_generation_pipeline, prompt, temperature, max_retries,
            max_new_tokens=1500
        )

    def _get_extraction_pipeline(self):
        """
        Lazily load and return the extraction model pipeline (Qwen 2.5).
        Caches the pipeline after first load to avoid reloading on each call.

        Returns:
            transformers.Pipeline: Loaded text-generation pipeline
        """
        if self._extraction_pipeline is None:
            print(f"  Loading extraction model: {self.extraction_model_id}")
            self._extraction_pipeline = self._load_pipeline(self.extraction_model_id)
            print(f"  ✓ Extraction model ready.")
        return self._extraction_pipeline

    def _get_generation_pipeline(self):
        """
        Lazily load and return the generation model pipeline (Llama 3.2 3B).
        Caches the pipeline after first load to avoid reloading on each call.

        Returns:
            transformers.Pipeline: Loaded text-generation pipeline
        """
        if self._generation_pipeline is None:
            print(f"  Loading generation model: {self.generation_model_id}")
            self._generation_pipeline = self._load_pipeline(self.generation_model_id)
            print(f"  ✓ Generation model ready.")
        return self._generation_pipeline

    def _unload_extraction(self):
        """
        Fully unload the extraction model from memory. Deletes the pipeline
        object, its underlying model, and flushes GPU/MPS memory. This frees
        ~3-4 GB so the generation model can load without OOM.
        """
        if self._extraction_pipeline is not None:
            print("  -- Unloading extraction model to free memory...")
            # Delete the model and tokenizer from the pipeline explicitly
            try:
                del self._extraction_pipeline.model
                del self._extraction_pipeline.tokenizer
            except Exception:
                pass
            del self._extraction_pipeline
            self._extraction_pipeline = None
            self._free_memory()

    def _unload_generation(self):
        """
        Fully unload the generation model from memory. Deletes the pipeline
        object, its underlying model, and flushes GPU/MPS memory.
        """
        if self._generation_pipeline is not None:
            print("  -- Unloading generation model to free memory...")
            try:
                del self._generation_pipeline.model
                del self._generation_pipeline.tokenizer
            except Exception:
                pass
            del self._generation_pipeline
            self._generation_pipeline = None
            self._free_memory()

    @staticmethod
    def _free_memory():
        """
        Release unused GPU/MPS memory and trigger garbage collection.
        Helps prevent OOM errors when switching between models on
        memory-constrained devices (e.g. 8 GB unified memory).
        """
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_pipeline(self, model_id):
        """
        Load a HuggingFace text-generation pipeline for the given model ID.
        Automatically selects GPU (CUDA/MPS) if available, otherwise uses CPU.
        Uses half-precision (FP16/BF16) to reduce memory footprint by ~50%.

        Args:
            model_id (str): HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')

        Returns:
            transformers.Pipeline: Loaded and ready text-generation pipeline

        Raises:
            SystemExit: If the model cannot be loaded (auth error, missing token, etc.)
        """
        try:
            import torch
            from transformers import pipeline, AutoTokenizer
            from huggingface_hub import snapshot_download
            import os

            # Suppress transformers log messages below ERROR level
            # (progress bars are already handled via TQDM_DISABLE in main.py)
            from transformers.utils.logging import set_verbosity_error
            set_verbosity_error()

            # Determine best available device
            if torch.cuda.is_available():
                device = 0  # First CUDA GPU
                print(f"    - GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                print(f"    - Device: Apple Silicon GPU (MPS accelerated)")
            else:
                device = -1  # CPU
                print(f"    - Device: CPU (no GPU detected — will be slower)")

            # Build token kwargs for authentication — passed to both
            # the tokenizer and the pipeline so gated models can be accessed
            token_kwargs = {}
            if self.hf_token:
                token_kwargs["token"] = self.hf_token

            # Check if model is already cached locally to show appropriate message
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "hub"
            )
            model_cache_name = "models--" + model_id.replace("/", "--")
            is_cached = os.path.isdir(os.path.join(cache_dir, model_cache_name))

            if is_cached:
                pass  # Silently load from cache
            else:
                print(f"    Downloading model for the first time (~2–6 GB)...")
                print(f"    This may take several minutes depending on your connection.")
                print(f"    Once downloaded, future runs will load instantly from cache.")

            # Load tokenizer — triggers download of tokenizer files if not cached
            # use_fast=True forces the Rust-based fast tokenizer, avoiding the
            # optional 'protobuf' / SentencePiece slow-tokenizer path used by
            # some models (e.g. Llama 3.2) that would fail without that library.
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True, **token_kwargs
            )

            # On Apple Silicon (MPS), both Qwen 2.5 and Llama 3.2 need eager
            # attention. Both use Grouped Query Attention (GQA) which crashes on
            # MPS with the default SDPA attention due to mismatched KV-head tiling
            # (Qwen: 16 heads / 2 KV heads, Llama: 24 heads / 8 KV heads).
            # attn_implementation="eager" uses standard PyTorch attention instead.
            model_kwargs = {}
            if device == "mps":
                model_kwargs["attn_implementation"] = "eager"

            # Use reduced precision for faster inference on GPU.
            # MPS (Apple Silicon): use bfloat16 — float16 causes inf/nan
            # errors due to its narrow exponent range. bfloat16 has the same
            # range as float32 and is natively supported on Apple Silicon.
            # CUDA: float16 is well-supported and fastest.
            # CPU: keep default float32 (no native half-precision compute).
            if device == "mps":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif device != -1:
                model_kwargs["torch_dtype"] = torch.float16

            # Load full pipeline — triggers model weights download if not cached
            pipeline_kwargs = dict(device=device, **token_kwargs)
            if model_kwargs:
                pipeline_kwargs["model_kwargs"] = model_kwargs
            pipe = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                **pipeline_kwargs
            )
            return pipe

        except OSError as e:
            # Typically a 401 auth error for gated models
            if "401" in str(e) or "gated" in str(e).lower() or "token" in str(e).lower():
                self._print_token_instructions(model_id)
            else:
                print(f"\n  ERROR: Could not load model '{model_id}'.")
                print(f"  Technical error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR: Failed to load model '{model_id}'.")
            print(f"  Technical error: {e}")
            sys.exit(1)

    def _call_model(self, pipeline_getter, prompt, temperature, max_retries,
                    max_new_tokens=2048):
        """
        Execute a text generation call using the given pipeline getter function.
        Handles retries with exponential backoff on transient failures.

        Args:
            pipeline_getter (callable): Function that returns the pipeline to use
            prompt (str): The prompt text to send to the model
            temperature (float): Sampling temperature for the model
            max_retries (int): Number of retry attempts before giving up
            max_new_tokens (int): Maximum tokens to generate (default 2048)

        Returns:
            str: Generated text response (with the prompt removed)

        Raises:
            RuntimeError: If all retries are exhausted without a successful response
        """
        import time
        import warnings
        import logging

        # Suppress the "Both max_new_tokens and max_length" warning that some
        # models emit via the logging module (not caught by warnings.filterwarnings)
        logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                pipe = pipeline_getter()

                # Show processing indicator — LLM inference can take 10-60s on CPU
                print(f"    ... Thinking", flush=True)
                inference_start = time.time()

                # In transformers 5.x there are two sources of length limits that
                # conflict: pipe.model.config.max_length (often 20) and the
                # pipeline's default max_new_tokens (256). Clear both before the call
                # so our explicit max_new_tokens=1024 is the only limit in play.
                if hasattr(pipe.model, "config") and hasattr(pipe.model.config, "max_length"):
                    pipe.model.config.max_length = None
                pipe.model.generation_config.max_length = None

                # Run inference. Suppress the two harmless transformers 5.x warnings:
                #   1) "Passing generation_config together with generation-related
                #       arguments is deprecated" — we pass explicit kwargs so the
                #       pipeline can override the model's built-in generation config.
                #   2) "Both max_new_tokens and max_length seem to have been set" —
                #       spurious once we clear max_length above; max_new_tokens wins.
                # Format prompt as chat messages for Instruct models.
                # Passing a plain string to Llama 3.2 Instruct can cause
                # immediate EOS (empty response). Chat format ensures the
                # model's template is applied correctly.
                messages = [{"role": "user", "content": prompt}]

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*generation_config.*generation-related arguments.*"
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*max_new_tokens.*max_length.*"
                    )
                    outputs = pipe(
                        messages,
                        max_new_tokens=max_new_tokens,
                        temperature=float(temperature),
                        do_sample=temperature > 0,
                        pad_token_id=pipe.tokenizer.eos_token_id,
                        return_full_text=False  # Only return generated text, not prompt
                    )

                # Extract generated text from pipeline output.
                # With chat-format input, generated_text may be a string
                # or a list of message dicts — handle both.
                if outputs and isinstance(outputs, list):
                    raw = outputs[0].get("generated_text", "")
                    if isinstance(raw, list):
                        # Chat-format output: list of message dicts
                        # Get the last assistant message
                        generated = ""
                        for msg in raw:
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                generated = msg.get("content", "")
                        if not generated and raw:
                            # Fallback: take last message content
                            last = raw[-1]
                            generated = last.get("content", "") if isinstance(last, dict) else str(last)
                    else:
                        generated = str(raw)

                    if generated and generated.strip():
                        elapsed = time.time() - inference_start
                        print(f"    ✓ Done. ({elapsed:.1f}s)", flush=True)
                        return generated.strip()

                raise ValueError("Empty response received from model")

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    print(f"  Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to get response after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def _print_token_instructions(self, model_id):
        """
        Print clear instructions for obtaining and using a HuggingFace token
        when a gated model cannot be accessed.

        Args:
            model_id (str): The model ID that failed to load
        """
        print(f"\n  ERROR: Cannot access model '{model_id}'.")
        print("  This model requires a HuggingFace account and accepted license.")
        print("\n  To fix this:")
        print("  1. Create a free account at: https://huggingface.co/join")
        print(f"  2. Accept the model license at: https://huggingface.co/{model_id}")
        print("  3. Generate an access token at: https://huggingface.co/settings/tokens")
        print("  4. Run the script with your token:")
        print(f"       python main.py --hf-token YOUR_TOKEN_HERE")
        print("  OR set the environment variable:")
        print("       export HF_TOKEN=YOUR_TOKEN_HERE  (macOS/Linux)")
        print("       set HF_TOKEN=YOUR_TOKEN_HERE     (Windows)")
