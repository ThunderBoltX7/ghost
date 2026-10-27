"""Codebase AST indexing, symbol lookup, search, project scanning, and project memory for Ghost CLI."""

import ast
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..ui import console, UI
from ..config import CACHE_FILE, GHOST_MD_FILE
from ..utils import add_xp, truncate_output
from ..git_manager import IGNORE_DIRS, _is_git_repo, _run_git

_EXT_LANG = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".rb": "Ruby",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ header",
    ".cs": "C#",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
}


class SymbolKind:
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    INTERFACE = "interface"
    STRUCT = "struct"


class SymbolItem:
    def __init__(
        self,
        name: str,
        kind: str,
        filepath: str,
        line: int = 1,
        col: int = 0,
        signature: str = "",
        docstring: str = "",
        parent: Optional[str] = None,
        bases: Optional[List[str]] = None,
    ):
        self.name = name
        self.kind = kind
        self.filepath = filepath
        self.line = line
        self.col = col
        self.signature = signature
        self.docstring = docstring
        self.parent = parent
        self.bases = bases or []

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "kind": self.kind,
            "filepath": self.filepath,
            "line": self.line,
        }
        if self.signature:
            d["signature"] = self.signature
        if self.parent:
            d["parent"] = self.parent
        if self.bases:
            d["bases"] = self.bases
        if self.docstring:
            d["docstring"] = self.docstring[:120]
        return d


class PythonAstIndexer(ast.NodeVisitor):
    def __init__(self, filepath: str, source_lines: List[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.symbols: List[SymbolItem] = []
        self.imports: List[Dict[str, Any]] = []
        self.references: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute) and hasattr(ast, "unparse"):
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    bases.append(b.attr)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)

        doc = ast.get_docstring(node) or ""
        sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        sym = SymbolItem(
            name=node.name,
            kind=SymbolKind.CLASS,
            filepath=self.filepath,
            line=node.lineno,
            col=node.col_offset,
            signature=sig,
            docstring=doc,
            parent=self.current_class,
            bases=bases,
        )
        self.symbols.append(sym)

        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_func(node, is_async=True)

    def _handle_func(self, node: Any, is_async: bool = False):
        doc = ast.get_docstring(node) or ""
        kind = SymbolKind.METHOD if self.current_class else SymbolKind.FUNCTION

        args_list = []
        for a in node.args.args:
            args_list.append(a.arg)
        if node.args.vararg:
            args_list.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args_list.append(f"**{node.args.kwarg.arg}")

        prefix = "async def " if is_async else "def "
        sig = f"{prefix}{node.name}({', '.join(args_list)})"
        if getattr(node, "returns", None) and hasattr(ast, "unparse"):
            try:
                sig += f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        sym = SymbolItem(
            name=node.name,
            kind=kind,
            filepath=self.filepath,
            line=node.lineno,
            col=node.col_offset,
            signature=sig,
            docstring=doc,
            parent=self.current_class,
        )
        self.symbols.append(sym)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.append({"module": alias.name, "name": name, "line": node.lineno})
            self.symbols.append(
                SymbolItem(
                    name=name,
                    kind=SymbolKind.IMPORT,
                    filepath=self.filepath,
                    line=node.lineno,
                    signature=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.append({"module": mod, "name": alias.name, "asname": alias.asname, "line": node.lineno})
            self.symbols.append(
                SymbolItem(
                    name=name,
                    kind=SymbolKind.IMPORT,
                    filepath=self.filepath,
                    line=node.lineno,
                    signature=f"from {mod} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                )
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.references.append({"symbol": node.id, "filepath": self.filepath, "line": node.lineno})
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        self.references.append({"symbol": node.attr, "filepath": self.filepath, "line": node.lineno})
        self.generic_visit(node)


def _python_search_code(pattern: str, root: str = ".", max_matches: int = 50) -> str:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    matches = []
    root_path = Path(root)
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz", ".whl", ".exe", ".bin", ".pyc"}:
                continue
            fp = Path(dirpath) / fn
            try:
                rel = str(fp.relative_to(root_path))
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{rel}:{line_no}:{line.strip()}")
                            if len(matches) >= max_matches:
                                break
            except Exception:
                continue
            if len(matches) >= max_matches:
                break
        if len(matches) >= max_matches:
            break

    return "\n".join(matches) if matches else "No matches."


def _python_search_code_structured(pattern: str, root: str = ".") -> List[Dict[str, Any]]:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    results = []
    root_path = Path(root)
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in _EXT_LANG or ext in (".json", ".md", ".toml", ".yaml", ".yml", ".txt"):
                fp = Path(dirpath) / fn
                try:
                    rel = str(fp.relative_to(root_path))
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for line_no, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append({
                                    "symbol": pattern,
                                    "filepath": rel,
                                    "line": line_no,
                                    "context": line.strip()[:140],
                                })
                                if len(results) >= 50:
                                    return results
                except Exception:
                    continue
    return results


class CodebaseIndex:
    def __init__(self, root: str = "."):
        self.root = Path(root).resolve()
        self.symbols: Dict[str, List[SymbolItem]] = {}
        self.definitions: Dict[str, List[SymbolItem]] = {}
        self.references: Dict[str, List[Dict[str, Any]]] = {}
        self.imports: Dict[str, List[Dict[str, Any]]] = {}
        self.classes: Dict[str, List[SymbolItem]] = {}
        self.file_symbols: Dict[str, List[SymbolItem]] = {}
        self.file_mtimes: Dict[str, float] = {}
        self.initialized = False

    def build_index(self, max_files: int = 2000):
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext in _EXT_LANG:
                    fp = os.path.relpath(os.path.join(dirpath, fn), str(self.root))
                    self.index_file(fp)
                    count += 1
                    if count >= max_files:
                        break
            if count >= max_files:
                break
        self.initialized = True

    def ensure_indexed(self):
        if not self.initialized:
            self.build_index()

    def remove_file(self, filepath: str):
        norm_path = os.path.normpath(filepath)
        old_symbols = self.file_symbols.pop(norm_path, [])
        for sym in old_symbols:
            key = sym.name.lower()
            if key in self.symbols:
                self.symbols[key] = [s for s in self.symbols[key] if os.path.normpath(s.filepath) != norm_path]
                if not self.symbols[key]:
                    del self.symbols[key]
            if key in self.definitions:
                self.definitions[key] = [s for s in self.definitions[key] if os.path.normpath(s.filepath) != norm_path]
                if not self.definitions[key]:
                    del self.definitions[key]
            if key in self.classes:
                self.classes[key] = [s for s in self.classes[key] if os.path.normpath(s.filepath) != norm_path]
                if not self.classes[key]:
                    del self.classes[key]

        self.imports.pop(norm_path, None)
        self.file_mtimes.pop(norm_path, None)
        for sym_name in list(self.references.keys()):
            self.references[sym_name] = [
                r for r in self.references[sym_name] if os.path.normpath(r.get("filepath", "")) != norm_path
            ]
            if not self.references[sym_name]:
                del self.references[sym_name]

    def index_file(self, filepath: str):
        norm_path = os.path.normpath(filepath)
        full_path = self.root / norm_path if not os.path.isabs(filepath) else Path(filepath)
        if not full_path.exists() or not full_path.is_file():
            self.remove_file(norm_path)
            return

        try:
            mtime = full_path.stat().st_mtime
            if norm_path in self.file_mtimes and self.file_mtimes[norm_path] == mtime:
                return
        except Exception:
            return

        self.remove_file(norm_path)

        try:
            size = full_path.stat().st_size
            if size > 500_000:
                return
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return

        self.file_mtimes[norm_path] = mtime
        ext = full_path.suffix.lower()

        if ext == ".py":
            self._index_python(norm_path, content)
        else:
            self._index_regex(norm_path, content, ext)

    def _index_python(self, filepath: str, content: str):
        lines = content.splitlines()
        try:
            tree = ast.parse(content, filename=filepath)
            indexer = PythonAstIndexer(filepath, lines)
            indexer.visit(tree)
            symbols = indexer.symbols
            imports = indexer.imports
            refs = indexer.references
        except Exception:
            self._index_regex(filepath, content, ".py")
            return

        self.file_symbols[filepath] = symbols
        self.imports[filepath] = imports

        for sym in symbols:
            key = sym.name.lower()
            self.symbols.setdefault(key, []).append(sym)
            if sym.kind in (SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.METHOD):
                self.definitions.setdefault(key, []).append(sym)
            if sym.kind == SymbolKind.CLASS:
                self.classes.setdefault(key, []).append(sym)

        for ref in refs:
            rkey = str(ref["symbol"]).lower()
            self.references.setdefault(rkey, []).append(ref)

    def _index_regex(self, filepath: str, content: str, ext: str):
        symbols = []
        imports = []
        lines = content.splitlines()

        patterns = [
            (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z0-9_$]+)(?:\s+extends\s+([A-Za-z0-9_$.]+))?", re.M), SymbolKind.CLASS),
            (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(([^)]*)\)", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*function\s*\(([^)]*)\)", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*def\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*class\s+([A-Za-z0-9_]+)(?:\(([^)]*)\))?:", re.M), SymbolKind.CLASS),
            (re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z0-9_]+)\s*\(", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z0-9_]+)", re.M), SymbolKind.STRUCT),
            (re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)\s*\(", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*type\s+([A-Za-z0-9_]+)\s+struct", re.M), SymbolKind.STRUCT),
            (re.compile(r"^\s*type\s+([A-Za-z0-9_]+)\s+interface", re.M), SymbolKind.INTERFACE),
        ]

        import_patterns = [
            re.compile(r"^\s*import\s+(?:\{([^}]+)\}|\*\s+as\s+([A-Za-z0-9_$]+)|([A-Za-z0-9_$]+))\s+from\s+['\"]([^'\"]+)['\"]", re.M),
            re.compile(r"^\s*(?:const|let|var)\s+(?:\{([^}]+)\}|([A-Za-z0-9_$]+))\s*=\s*require\(['\"]([^'\"]+)['\"]\)", re.M),
            re.compile(r"^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)|import\s+([A-Za-z0-9_., ]+))", re.M),
        ]

        for line_no, line in enumerate(lines, 1):
            for pat, kind in patterns:
                m = pat.match(line)
                if m:
                    name = m.group(1)
                    sig = line.strip().rstrip("{: ")
                    sym = SymbolItem(name=name, kind=kind, filepath=filepath, line=line_no, signature=sig)
                    symbols.append(sym)
                    break
            for ipat in import_patterns:
                im = ipat.match(line)
                if im:
                    imports.append({"line": line_no, "text": line.strip()})
                    break

        self.file_symbols[filepath] = symbols
        self.imports[filepath] = imports
        for sym in symbols:
            key = sym.name.lower()
            self.symbols.setdefault(key, []).append(sym)
            if sym.kind in (SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.STRUCT, SymbolKind.INTERFACE):
                self.definitions.setdefault(key, []).append(sym)
            if sym.kind == SymbolKind.CLASS:
                self.classes.setdefault(key, []).append(sym)

    def invalidate_file(self, filepath: str):
        norm_path = os.path.normpath(filepath)
        self.index_file(norm_path)

    def find_symbol(self, query: str, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure_indexed()
        query_low = query.strip().lower()
        results = []
        for key, sym_list in self.symbols.items():
            if query_low in key or query_low == key:
                for sym in sym_list:
                    if kind and sym.kind.lower() != kind.lower():
                        continue
                    results.append(sym.to_dict())
        return sorted(results, key=lambda x: (0 if x["name"].lower() == query_low else 1, x["filepath"], x["line"]))[:50]

    def find_definition(self, symbol: str, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure_indexed()
        sym_low = symbol.strip().lower()
        matches = self.definitions.get(sym_low, [])
        if not matches:
            return self.find_symbol(symbol)
        results = [m.to_dict() for m in matches]
        if filepath:
            norm_f = os.path.normpath(filepath)
            results.sort(key=lambda x: 0 if os.path.normpath(x["filepath"]) == norm_f else 1)
        return results[:30]

    def find_references(self, symbol: str) -> List[Dict[str, Any]]:
        self.ensure_indexed()
        sym_low = symbol.strip().lower()
        refs = self.references.get(sym_low, [])
        if refs:
            return refs[:50]
        return _python_search_code_structured(symbol, str(self.root))[:50]

    def find_related_files(self, filepath: str) -> List[Dict[str, str]]:
        self.ensure_indexed()
        norm_target = os.path.normpath(filepath)
        target_name = Path(norm_target).stem.lower()
        target_dir = os.path.dirname(norm_target)
        related = {}

        target_imports = self.imports.get(norm_target, [])
        for imp in target_imports:
            mod = imp.get("module") or imp.get("text", "")
            for f in self.file_symbols:
                if f != norm_target and (Path(f).stem.lower() in mod.lower() or mod.lower() in f.lower()):
                    related[f] = f"Imported by {norm_target}"

        for f, imp_list in self.imports.items():
            if f == norm_target:
                continue
            for imp in imp_list:
                mod = imp.get("module") or imp.get("text", "")
                if target_name in mod.lower() or norm_target.replace("\\", "/").lower() in mod.lower():
                    related[f] = f"Imports {norm_target}"

        for f in self.file_symbols:
            if f != norm_target and os.path.dirname(f) == target_dir:
                related.setdefault(f, "Same directory")

        target_syms = {s.name.lower() for s in self.file_symbols.get(norm_target, [])}
        for f, syms in self.file_symbols.items():
            if f != norm_target:
                for s in syms:
                    for b in s.bases:
                        if b.lower() in target_syms:
                            related[f] = f"Inherits {b} from {norm_target}"

        return [{"filepath": k, "reason": v} for k, v in list(related.items())[:25]]


INDEX = CodebaseIndex()


def _detect_languages(files):
    counts = {}
    for f in files:
        lang = _EXT_LANG.get(Path(f).suffix)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def scan_project(root=".", max_depth=3, max_files=2000):
    root_path = Path(root)
    files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        rel = Path(dirpath).relative_to(root_path)
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue
        for fn in filenames:
            files.append(str((rel / fn)))
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break

    languages = _detect_languages(files)
    markers = {
        "git": (root_path / ".git").exists(),
        "package.json": (root_path / "package.json").exists(),
        "pyproject.toml": (root_path / "pyproject.toml").exists(),
        "requirements.txt": (root_path / "requirements.txt").exists(),
        "Dockerfile": (root_path / "Dockerfile").exists(),
        "GHOST.md": (root_path / "GHOST.md").exists(),
    }
    top_level = sorted({f.split(os.sep)[0] for f in files if f})[:50]

    return {
        "root": str(root_path.resolve()),
        "file_count": len(files),
        "languages": languages,
        "markers": markers,
        "top_level": top_level,
        "scanned_at": time.time(),
    }


CACHE_TTL_SECONDS = 300


def _load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(data):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def get_cached_scan(root, scanner_fn, force=False):
    cache = _load_cache()
    key = str(Path(root).resolve())
    entry = cache.get(key)
    now = time.time()

    if not force and entry and (now - entry.get("scanned_at", 0)) < CACHE_TTL_SECONDS:
        return entry, True

    fresh = scanner_fn(root)
    cache[key] = fresh
    _save_cache(cache)
    return fresh, False


def invalidate_cache(root="."):
    cache = _load_cache()
    key = str(Path(root).resolve())
    if key in cache:
        del cache[key]
        _save_cache(cache)


def load_ghost_md() -> str:
    if GHOST_MD_FILE.exists():
        try:
            content = GHOST_MD_FILE.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return f"\n\n--- Project Memory (GHOST.md) ---\n{content[:6000]}\n---"
        except Exception:
            pass
    return ""


def generate_ghost_md() -> str:
    UI.status("Analyzing project to generate GHOST.md...")
    scan_res, _ = get_cached_scan(".", scan_project, force=True)
    from ..verification import detect_test_commands
    test_cmds = detect_test_commands(".")
    langs = ", ".join(f"{k} ({v} files)" for k, v in scan_res["languages"].items()) or "Unknown"
    tests_str = "\n".join(f"- **{lang}**: `{' '.join(cmd)}`" for lang, cmd in test_cmds) or "- No automated test suite detected yet."

    linters = []
    for tool in ("ruff", "mypy", "pytest", "eslint", "tsc"):
        if shutil.which(tool):
            linters.append(tool)

    template = f"""# GHOST.md — Project Memory & Conventions

## Overview
- **Root**: `{scan_res['root']}`
- **Primary Languages**: {langs}
- **Top-Level Structure**: `{', '.join(scan_res['top_level'][:25])}`

## Commands
### Testing
{tests_str}

### Linting & Type Checking
- Available on PATH: `{', '.join(linters) if linters else 'none detected'}`

## Code Style & Conventions
- Read target files before applying edits.
- Keep changes minimal, modular, and well-tested.
- Preserve existing line endings and formatting conventions.

## Architecture & Gotchas
- Add project-specific notes, domain invariants, or tricky edge cases here.
"""
    GHOST_MD_FILE.write_text(template, encoding="utf-8")
    UI.success(f"Generated {GHOST_MD_FILE.resolve()}")
    return template


def _scan_project_tool() -> str:
    UI.status("Scanning project...")
    result, cached = get_cached_scan(".", scan_project)
    lang_summary = ", ".join(f"{k} ({v})" for k, v in result["languages"].items()) if result["languages"] else "none detected"
    UI.success(f"Scanned {result['file_count']} files ({lang_summary}){' [cached]' if cached else ''}")
    if UI.verbose:
        from ..ui import _print_scan
        _print_scan(result, cached)
    lines = [f"Root: {result['root']} (cache {'hit' if cached else 'miss'})",
             f"Files: {result['file_count']}"]
    if result["languages"]:
        lines.append("Languages: " + lang_summary)
    markers = [k for k, v in result["markers"].items() if v]
    if markers:
        lines.append("Markers: " + ", ".join(markers))
    lines.append("Top-level: " + ", ".join(result["top_level"][:20]))
    return "\n".join(lines)


# Need shutil for generate_ghost_md
import shutil
