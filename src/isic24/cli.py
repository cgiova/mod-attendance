from .paths import REPO_ROOT, DATA_DIR, MODELS_DIR

def main():
    print("repo:", REPO_ROOT)
    print("data:", DATA_DIR.exists(), DATA_DIR)
    print("models:", MODELS_DIR.exists(), MODELS_DIR)

if __name__ == "__main__":
    main()