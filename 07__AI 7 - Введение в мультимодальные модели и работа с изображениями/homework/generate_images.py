"""Сравнение исходных и уточнённых промптов Stable Diffusion."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен; используйте --device cpu")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.enable_attention_slicing()
    cases = json.loads((ROOT / "prompts.json").read_text())
    (ROOT / "images").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    records = []
    for case in cases:
        for variant in ["base", "improved"]:
            negative = "" if variant == "base" else "blurry, low quality, text, watermark"
            start = time.monotonic()
            # Новый генератор с тем же seed для каждой пары изображений.
            output = pipe(
                prompt=case[variant], negative_prompt=negative,
                generator=torch.Generator(device="cpu").manual_seed(case["seed"]),
                num_inference_steps=args.steps, guidance_scale=7.5,
                height=512, width=512,
            )
            name = case["id"] + "_" + variant + ".png"
            output.images[0].save(ROOT / "images" / name)
            record = dict(id=case["id"], variant=variant, prompt=case[variant],
                          negative_prompt=negative, seed=case["seed"], model=model_id,
                          revision=getattr(pipe.config, "_commit_hash", None),
                          scheduler=type(pipe.scheduler).__name__, steps=args.steps,
                          guidance_scale=7.5, width=512, height=512, device=device,
                          dtype="float32", seconds=round(time.monotonic()-start, 2),
                          nsfw_detected=output.nsfw_content_detected,
                          image="images/"+name)
            records.append(record)
            (ROOT / "results" / "generation.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2))
            print("Сохранено", name, flush=True)


if __name__ == "__main__":
    main()
