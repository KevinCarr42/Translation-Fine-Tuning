# find the base model
from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
BASE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

for repo in cache_info.repos:
    if BASE_MODEL_ID in repo.repo_id:
        print(f"Found: {repo.repo_id}")
        print(f"Path: {repo.repo_path}")
        print(f"Size: {repo.size_on_disk / 1e9:.2f} GB")
        print(f"Revisions: {[r.commit_hash for r in repo.revisions]}")


# Check which revision is the default
from huggingface_hub import list_repo_refs

refs = list_repo_refs(BASE_MODEL_ID)
print(f"Default branch: {refs.branches[0].name if refs.branches else 'main'}")
print(f"Latest commit: {refs.branches[0].target_commit if refs.branches else 'unknown'}")

