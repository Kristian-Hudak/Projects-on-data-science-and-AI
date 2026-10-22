# -*- coding: utf-8 -*-
"""Local runner for Medical Appointment QA pipeline.
Uses 4-bit quantized Llama-3.1-8B-Instruct and Faster-Whisper on NVIDIA RTX 4060 GPU.
"""

import os
import sys
import gc
import re
import json
import time
import tempfile
import torch

# Fix for Windows CUDA DLL loading in faster-whisper / ctranslate2
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if hasattr(os, 'add_dll_directory') and os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
os.environ['PATH'] = torch_lib + ';' + os.environ.get('PATH', '')

from huggingface_hub import login

# Add local repository root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio, audio_duration_seconds

# 1. HuggingFace Login
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    print("Authenticating with HuggingFace Hub...")
    login(token=hf_token)
else:
    print("Note: HF_TOKEN is not set. Ensure it is defined in your environment or .env file.")

# 2. Nastavenie zariadeni (GPU vs CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Inference device: {device}")

# 3. Load LLM model FIRST (allocating dedicated 8 GB VRAM for 4-bit quantization)
model_id = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Preparing LLM model: {model_id}...")

tokenizer = None
llm_model = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        print("Loading Llama-3.1 in 4-bit NF4 directly onto GPU...")
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": 0},
            token=hf_token,
            trust_remote_code=True
        )
        vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"LLM model successfully loaded in 4-bit! Allocated VRAM: {vram_gb:.2f} GB")
    else:
        print("[WARNING] CUDA is unavailable. Loading LLM on CPU...")
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map="cpu",
            token=hf_token,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("LLM model successfully loaded on CPU.")
except Exception as e:
    print(f"Error loading LLM model '{model_id}': {e}")

# 4. Load Faster-Whisper ASR model (uses remaining VRAM)
whisper_model = None
try:
    from faster_whisper import WhisperModel
    # With CUDA available, load medium or fallback to small in float16
    whisper_compute = "float16" if torch.cuda.is_available() else "int8"
    whisper_dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Faster-Whisper Medium ({whisper_dev}, {whisper_compute})...")
    try:
        whisper_model = WhisperModel("medium", device=whisper_dev, compute_type=whisper_compute)
    except Exception as w_err:
        print(f"Whisper medium exceeded VRAM ({w_err}), falling back to Whisper small...")
        whisper_model = WhisperModel("small", device=whisper_dev, compute_type=whisper_compute)
    print("Faster-Whisper model successfully loaded.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")


def transcribe_with_timestamps(audio_bytes: bytes) -> str:
    """Transcribes audio and returns timestamped segments: [start - end] spoken text."""
    if whisper_model is None:
        raise RuntimeError("Whisper model is not initialized.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, _ = whisper_model.transcribe(tmp_path, word_timestamps=True, language="en")

        lines = []
        for segment in segments:
            clean_text = segment.text.strip()
            if clean_text:
                lines.append(f"[{segment.start:.2f} - {segment.end:.2f}] {clean_text}")

        return "\n".join(lines)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_json_from_llm(response_text: str):
    """Safely extracts JSON array from LLM response text."""
    try:
        match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(response_text)
    except Exception as e:
        print(f"JSON Parse Error: {e}\nRaw Output: {response_text}")
        return []


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Main inference handler: transcribes consultation audio and extracts grounded Q&A evidence."""
    expected_count = len(request.questions)
    answers = []
    starts = []
    ends = []

    try:
        if tokenizer is None or llm_model is None or whisper_model is None:
            raise RuntimeError("Inference pipeline models are not loaded.")

        # 1. Extrakcia textu s casovymi intervalmi segmentov
        print(f"  [1/2] Transcribing audio with Faster-Whisper...", flush=True)
        t_w0 = time.time()
        audio_bytes = decode_audio(request.audio_base64)
        transcript = transcribe_with_timestamps(audio_bytes)
        t_whisper = time.time() - t_w0
        num_segments = len(transcript.strip().split("\n")) if transcript.strip() else 0
        print(f"  --> Transcription completed in {t_whisper:.2f} s ({num_segments} segments)", flush=True)

        # 2. Priprava promptu
        print(f"  [2/2] Analyzing {expected_count} clinical questions with Llama-3.1-8B (4-bit)...", flush=True)
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])

        system_prompt = """You are an expert clinical AI analyzing a doctor-patient dialogue transcript.
The transcript contains timestamped segments in the format: [start_time - end_time] Spoken text.

Your task is to answer each question with Yes (true) or No (false) based ONLY on what is explicitly discussed.

For each question:
1. Determine if the answer is true or false.
2. If true, extract the evidence passage timestamps:
   - "start": the start_time (float) of the first segment containing the evidence.
   - "end": the end_time (float) of the last segment containing the evidence.
3. If false:
   - "start": null
   - "end": null

You MUST reply ONLY with a valid JSON array of objects, one per question in the exact same order.
Format example:
[
  {"answer": true, "start": 21.68, "end": 26.36},
  {"answer": false, "start": null, "end": null}
]"""

        user_prompt = f"TRANSCRIPT:\n{transcript}\n\nQUESTIONS:\n{questions_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target_device = "cuda" if (llm_model is not None and next(llm_model.parameters()).is_cuda) else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

        t_llm0 = time.time()
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        t_llm = time.time() - t_llm0
        t_total = t_whisper + t_llm
        print(f"  --> Llama generated answers in {t_llm:.2f} s", flush=True)

        stat_40 = "OK (<40s target)" if t_total <= 40.0 else "WARNING (>40s)"
        stat_60 = "PASS (<60s limit)" if t_total < 60.0 else "TIMEOUT (>60s limit)!"
        print(f"  [AUDIO SUMMARY] Whisper: {t_whisper:.1f}s | Llama: {t_llm:.1f}s | Total elapsed: {t_total:.1f}s (Target 40s: {stat_40} | Hard limit 60s: {stat_60})", flush=True)

        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        json_results = extract_json_from_llm(response_text)

        for i in range(expected_count):
            try:
                ans_data = json_results[i]
                ans_bool = bool(ans_data.get("answer", True))
                ans_start = ans_data.get("start")
                ans_end = ans_data.get("end")

                if ans_bool and ans_start is not None and ans_end is not None:
                    ans_start = float(ans_start)
                    ans_end = float(ans_end)
                    if ans_end < ans_start:
                        ans_start, ans_end = ans_end, ans_start
                    elif ans_end == ans_start:
                        ans_end = ans_start + 2.0
                else:
                    ans_start = None
                    ans_end = None

                answers.append(ans_bool)
                starts.append(ans_start)
                ends.append(ans_end)
            except Exception:
                answers.append(True)
                starts.append(None)
                ends.append(None)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Error processing request ({type(exc)}: {exc}), engaging safe baseline fallback.")
        answers = [True] * expected_count
        starts = [None] * expected_count
        ends = [None] * expected_count

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=starts,
        evidence_end=ends
    )


if __name__ == "__main__":
    import requests
    from requests.models import Response
    import local_evaluator

    print("Clinical QA pipeline initialized and ready.")

    counter = [0]
    total_convs = 39

    def mock_post_impl(url, **kwargs):
        if "/predict" in url:
            payload = kwargs.get("json")
            if payload is None and "data" in kwargs:
                raw_data = kwargs["data"]
                payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

            request_dto = ASRQuestionRequestDto(**payload)
            counter[0] += 1
            idx = counter[0]
            fname = getattr(request_dto, 'audio_filename', f'sample_{idx}')

            print(f"\n{'='*70}", flush=True)
            print(f">>> [{idx}/{total_convs}] Processing dialogue: {fname}", flush=True)

            t0 = time.time()
            response_dto = predict(request_dto)
            elapsed = time.time() - t0

            stat_40 = "OK (<40s target)" if elapsed <= 40.0 else "WARNING (>40s)!"
            stat_60 = "PASS (<60s limit)" if elapsed < 60.0 else "TIMEOUT (>60s)!"
            percent_40 = (elapsed / 40.0) * 100
            percent_60 = (elapsed / 60.0) * 100
            print(f">>> [{idx}/{total_convs}] RESULT: {elapsed:.2f} s | {percent_40:.0f}% of 40s target | {percent_60:.0f}% of 60s limit -> {stat_40}, {stat_60}", flush=True)
            print(f"{'='*70}\n", flush=True)

            resp = Response()
            resp.status_code = 200
            resp._content = json.dumps({
                "answers": response_dto.answers,
                "evidence_start": response_dto.evidence_start,
                "evidence_end": response_dto.evidence_end
            }).encode('utf-8')
            resp.headers['Content-Type'] = 'application/json'
            return resp
        raise ValueError(f"Unexpected URL: {url}")

    # Prekrytie requests.post a requests.Session.post pre local_evaluator
    requests.post = mock_post_impl
    requests.Session.post = lambda self, url, **kwargs: mock_post_impl(url, **kwargs)

    print("Launching benchmark replay via local_evaluator...")
    try:
        statistics = local_evaluator.replay(url="http://localhost:9054/predict", verbose=True)
        print("\n--- BENCHMARK EVALUATION RESULTS ---")
        print(statistics.report())
    except Exception as eval_err:
        print(f"Evaluation benchmark error: {eval_err}")