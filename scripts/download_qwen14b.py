from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen3-14B"

def main():
    path = snapshot_download(
        repo_id=MODEL_NAME,
        resume_download=True,
    )
    print(f"Downloaded model to: {path}")


if __name__ == "__main__":
    main()
