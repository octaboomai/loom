"""
loom.context.repo_map
-----------------------
Builds a compact "map" of a repository: for every source file, the
top-level symbols (functions/classes/etc.) it defines, extracted via
tree-sitter when a grammar is available and falling back to a regex
heuristic otherwise (so this never hard-fails on an unsupported language).

This map is what gets fed to the Planner/Coder agents instead of raw
file contents — keeps context small and relevant, per the "shallow repo
understanding" gap called out in the design doc.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

IGNORE_DIRS = {
    ".git", ".loom", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox", "target",
}

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".rs": "rust", ".java": "java", ".rb": "ruby", ".c": "c", ".cpp": "cpp",
}

# Fallback regexes per language: (kind, pattern) — used when tree-sitter
# grammar isn't installed/available for that language.
FALLBACK_PATTERNS = {
    "python": [("def", r"^\s*def\s+(\w+)\s*\("), ("class", r"^\s*class\s+(\w+)")],
    "javascript": [
        ("function", r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("),
        ("class", r"^\s*(?:export\s+)?class\s+(\w+)"),
        ("const_fn", r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\("),
    ],
    "typescript": [
        ("function", r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("),
        ("class", r"^\s*(?:export\s+)?class\s+(\w+)"),
        ("interface", r"^\s*(?:export\s+)?interface\s+(\w+)"),
    ],
    "go": [("func", r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*\("), ("type", r"^\s*type\s+(\w+)")],
    "rust": [("fn", r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\("), ("struct", r"^\s*(?:pub\s+)?struct\s+(\w+)")],
    "java": [("method", r"^\s*(?:public|private|protected)[^;{]*\s(\w+)\s*\("), ("class", r"^\s*(?:public\s+)?class\s+(\w+)")],
    "ruby": [("def", r"^\s*def\s+(\w+)"), ("class", r"^\s*class\s+(\w+)")],
    "c": [("func", r"^\w[\w\s\*]*\s(\w+)\s*\([^;]*\)\s*\{")],
    "cpp": [("func", r"^\w[\w\s\*:<>]*\s(\w+)\s*\([^;]*\)\s*\{"), ("class", r"^\s*class\s+(\w+)")],
}


@dataclass
class Symbol:
    kind: str
    name: str
    line: int


@dataclass
class FileMap:
    path: str
    lang: str
    symbols: list[Symbol] = field(default_factory=list)
    loc: int = 0

    def render(self) -> str:
        if not self.symbols:
            return f"{self.path}"
        sig = ", ".join(f"{s.kind} {s.name}" for s in self.symbols[:40])
        more = "" if len(self.symbols) <= 40 else f" (+{len(self.symbols)-40} more)"
        return f"{self.path} [{self.loc} loc]: {sig}{more}"


def _try_tree_sitter(path: Path, lang: str, text: str) -> list[Symbol] | None:
    try:
        from tree_sitter_languages import get_parser
    except Exception:
        return None
    try:
        parser = get_parser(lang)
    except Exception:
        return None

    text_bytes = text.encode("utf-8")
    tree = parser.parse(text_bytes)
    symbols: list[Symbol] = []

    def_kinds = {
        "function_definition", "class_definition", "method_definition",
        "function_declaration", "class_declaration", "interface_declaration",
        "struct_item", "impl_item",
    }

    def walk(node):
        if node.type in def_kinds:
            name_node = None
            for child in node.children:
                if child.type in ("identifier", "type_identifier", "property_identifier"):
                    name_node = child
                    break
            # Slice on the UTF-8 BYTES (tree-sitter offsets are byte offsets,
            # not char offsets) then decode — slicing `text` directly breaks
            # on any multi-byte character earlier in the file.
            name = (text_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                    if name_node else "?")
            symbols.append(Symbol(kind=node.type.replace("_definition", "").replace("_declaration", ""),
                                   name=name, line=node.start_point[0] + 1))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return symbols


def _fallback_regex(lang: str, text: str) -> list[Symbol]:
    patterns = FALLBACK_PATTERNS.get(lang, [])
    symbols: list[Symbol] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for kind, pat in patterns:
            m = re.match(pat, line)
            if m:
                symbols.append(Symbol(kind=kind, name=m.group(1), line=i))
    return symbols


def map_file(path: Path) -> FileMap | None:
    ext = path.suffix.lower()
    lang = EXT_LANG.get(ext)
    if not lang:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    symbols = _try_tree_sitter(path, lang, text)
    if symbols is None:
        symbols = _fallback_regex(lang, text)
    return FileMap(path=str(path), lang=lang, symbols=symbols, loc=text.count("\n") + 1)


def build_repo_map(root: Path, max_files: int = 800) -> list[FileMap]:
    maps: list[FileMap] = []
    count = 0
    for p in sorted(root.rglob("*")):
        if count >= max_files:
            break
        if not p.is_file():
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        fm = map_file(p)
        if fm:
            maps.append(fm)
            count += 1
    return maps


def rank_relevant_files(maps: list[FileMap], query: str, top_k: int = 15) -> list[FileMap]:
    """Cheap relevance ranking: token overlap between query and
    (path + symbol names). No embeddings/vector DB dependency required —
    good enough to shrink context before an LLM call, which is the point.
    """
    q_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.lower()))
    scored = []
    for fm in maps:
        hay = fm.path.lower() + " " + " ".join(s.name.lower() for s in fm.symbols)
        hay_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", hay))
        score = len(q_tokens & hay_tokens)
        # small boost for shorter paths (usually more central files)
        score += 0.01 * max(0, 30 - len(fm.path))
        scored.append((score, fm))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [fm for score, fm in scored[:top_k] if score > 0] or [fm for _, fm in scored[:5]]


def render_map(maps: list[FileMap]) -> str:
    return "\n".join(fm.render() for fm in maps)
