import modal

app = modal.App(
    "pdf-ocr-jobs"
) 


CACHE_PATH = "/root/model_cache"
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"


def download_model_weights() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_NAME, cache_dir=CACHE_PATH)


image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "donut-python==1.0.7",
        "huggingface-hub==0.16.4",
        "transformers==4.21.3",
        "timm==0.5.4",
    )
    .run_function(download_model_weights)
)


@app.function(
    gpu="any",
    image=image,
    retries=3,
)
def parse_receipt(image: bytes):
    import io

    import torch
    from donut import DonutModel
    from PIL import Image

    # Use donut fine-tuned on an OCR dataset.
    task_prompt = "<s_cord-v2>"
    pretrained_model = DonutModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_PATH,
    )

    # Initialize model.
    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device)

    # Run inference.
    input_img = Image.open(io.BytesIO(image))
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)[
        "predictions"
    ][0]
    print("Result: ", output)

    return output


import pypdfium2 # Needs to be at the top to avoid warnings
import argparse
import os

from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models

from marker.output import save_markdown

configure_logging()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="PDF file to parse")
    parser.add_argument("output", help="Output base folder path")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to parse")
    parser.add_argument("--start_page", type=int, default=None, help="Page to start processing at")
    parser.add_argument("--langs", type=str, help="Languages to use for OCR, comma separated", default=None)
    parser.add_argument("--batch_multiplier", type=int, default=2, help="How much to increase batch sizes")
    args = parser.parse_args()

    langs = args.langs.split(",") if args.langs else None

    fname = args.filename
    model_lst = load_all_models()
    full_text, images, out_meta = convert_single_pdf(fname, model_lst, max_pages=args.max_pages, langs=langs, batch_multiplier=args.batch_multiplier, start_page=args.start_page)

    fname = os.path.basename(fname)
    subfolder_path = save_markdown(args.output, fname, full_text, images, out_meta)

    print(f"Saved markdown to the {subfolder_path} folder")


if __name__ == "__main__":
    main()

@app.local_entrypoint()
def main():
    from pathlib import Path

    receipt_filename = Path(__file__).parent / "receipt_00014.png"
    if receipt_filename.exists():
        with open(receipt_filename, "rb") as f:
            image = f.read()
    else:
        image = urllib.request.urlopen(
            "https://nwlc.org/wp-content/uploads/2022/01/Brandys-walmart-receipt-8.webp"
        ).read()
    print(parse_receipt.remote(image))

# poetry run modal run --env=main modal_serverless::app.main