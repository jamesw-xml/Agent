import os, json, io

ROOT = r"C:\code\Remundo.Pricing.Europe"
OUT  = "code_dataset.jsonl"

# what to include
INCLUDE_EXT = {
    ".cs",".ts",".tsx",".js",".jsx",".py",".java",".go",".rb",
    ".yml",".yaml",".md",".proto",".graphql",".sql",".tf",
    ".json",".toml",".ini",".sh",".ps1",".dockerfile",".gradle"
}

# what to skip
SKIP_DIRS = {
    ".git","node_modules",".idea",".vscode","dist","build","out","bin","obj",
    "__pycache__",".venv","venv",".terraform",".next",".svelte-kit","coverage",
    "agent"
}
SKIP_FILES = {"package-lock.json","pnpm-lock.yaml","yarn.lock",".DS_Store"}

# basic guards
MAX_FILE_BYTES = 2_000_000   # 2MB per file
MAX_LINE_LEN   = 6000        # drop insanely long lines (minified blobs)

# chunking
MAX_LINES   = 220            # ~few hundred tokens depending on code
OVERLAP     = 40             # keep context across chunks

def is_probably_text(path, sniff=2048):
    try:
        with open(path, "rb") as fh:
            b = fh.read(sniff)
        if not b:
            return False
        # null bytes => likely binary
        return b.find(b"\x00") == -1
    except Exception:
        return False

def iter_code_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune directories in-place
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]

        for f in filenames:
            if f in SKIP_FILES:
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in INCLUDE_EXT:
                continue
            path = os.path.join(dirpath, f)
            yield path

def chunk_lines(lines, max_lines=MAX_LINES, overlap=OVERLAP):
    i = 0
    n = len(lines)
    while i < n:
        j = min(n, i + max_lines)
        yield lines[i:j]
        if j >= n:
            break
        i = j - overlap if j - overlap > i else j

def relpath(root, path):
    rp = os.path.relpath(path, root).replace("\\", "/")
    parts = rp.split("/")
    service = parts[0] if parts else ""
    return rp, service

def main():
    total = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for path in iter_code_files(ROOT):
            try:
                if os.path.getsize(path) > MAX_FILE_BYTES:
                    continue
                if not is_probably_text(path):
                    continue

                with io.open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    raw = fh.read()
                if not raw.strip():
                    continue

                # drop absurdly long lines (minified artifacts)
                safe_lines = [ln for ln in raw.splitlines() if len(ln) <= MAX_LINE_LEN]
                if not safe_lines:
                    continue

                rp, service = relpath(ROOT, path)

                # add a small header per chunk so the model keeps context
                header = f"// file: {rp}\n// service: {service}\n"
                for lines in chunk_lines(safe_lines):
                    text = header + "\n".join(lines).strip()
                    if len(text) < 40:
                        continue
                    rec = {"text": text}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
                    print(f"Wrote chunk {total} from {rp}")

            except Exception as e:
                print("Skipping", path, e)

    print(f"Wrote {total} chunks to {OUT}")

if __name__ == "__main__":
    main()
