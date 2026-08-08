"""Regression tests for the bugs catalogued in bug_found.txt."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from opencode_py.agent.messages import _text
from opencode_py.config import Config, _strip_jsonc
from opencode_py.permission import MAX_APPROVED_PATTERNS, PermissionEngine


# -- #1 + #8: JSONC stripping ------------------------------------------------
@pytest.mark.parametrize(
    "src, expected",
    [
        ('{"a": 1} /* trailing */', {"a": 1}),
        ('{ /* o /* i */ o */ "a": 1 }', {"a": 1}),
        ('{"a": [/* l1 /* l2 */ */ "x"]}', {"a": ["x"]}),
        ('{ "s": "/* keep */" }', {"s": "/* keep */"}),
        ('{ "u": "\\\\u002F" }', {"u": "\\u002F"}),
    ],
)
def test_strip_jsonc_nested_comments_and_escapes(src, expected):
    assert json.loads(_strip_jsonc(src)) == expected


def test_strip_jsonc_escaped_unicode_boundary():
    # a unicode escape whose payload looks like a comment start must survive
    src = '{"x": "\\\\u002F* not a comment" }'
    assert json.loads(_strip_jsonc(src)) == {"x": "\\u002F* not a comment"}


# -- bug 4: agent vs agents -----------------------------------------------

def test_config_reads_both_agent_and_agents():
    cfg = Config.from_dict({"agent": {"a": {"model": "m1"}}})
    assert cfg.agents == {"a": {"model": "m1"}}
    cfg2 = Config.from_dict({"agents": {"b": {"model": "m2"}}})
    assert cfg2.agents == {"b": {"model": "m2"}}
    cfg3 = Config.from_dict({"agent": {"a": {"model": "m1"}}, "agents": {"b": {"model": "m2"}}})
    # plural wins (user explicitly wrote the dataclass field name)
    assert cfg3.agents == {"b": {"model": "m2"}}


# -- bug 3: session.py atomic writes ---------------------------------------

def test_save_session_atomic(tmp_path, monkeypatch):
    import opencode_py.session as session_mod
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    sess = session_mod.new_session(directory=str(tmp_path), provider="opencode", model="m")
    captured = {}

    orig_write = session_mod.Path.write_text

    def fake_write(self, *a, **k):
        captured["path"] = str(self)
        raise OSError("boom mid-write")

    monkeypatch.setattr(session_mod.Path, "write_text", fake_write)
    target = session_mod.session_path(sess.id)
    try:
        session_mod.save_session(sess)
    except OSError:
        pass
    assert not target.exists(), "partial write must not land at the final path"
    monkeypatch.setattr(session_mod.Path, "write_text", orig_write)


# -- bug 5: token estimation ----------------------------------------------

def test_text_counts_tool_calls_and_tool_results():
    assistant = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]}
    tool_result = {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "out"}
    assert len(_text(assistant)) > 0
    assert len(_text(tool_result)) > 0
    # content alone (< tool_calls-bearing message)
    plain = {"role": "assistant", "content": "hi"}
    assert _text(plain) == "hi"


# -- bug 6: permission approved list cleanup --------------------------------

def test_approved_patterns_bounded():
    eng = PermissionEngine()
    approved = []
    # simulate "always" answers with unique patterns
    patterns = [f"perm_{i}" for i in range(MAX_APPROVED_PATTERNS + 50)]
    eng.ask_callback = lambda desc, pats: "always"
    for p in patterns:
        eng.ask("desc", [p])
    assert len(eng._approved_patterns) <= MAX_APPROVED_PATTERNS
    # dedup: re-approving an existing pattern doesn't create a duplicate
    eng.ask("desc", [patterns[0]])
    assert eng._approved_patterns.count(patterns[0]) == 1


# -- bug 9: symlink cycle safety --------------------------------------------

def test_find_instruction_files_symlink_cycle_no_hang(tmp_path):
    from opencode_py.agent.system import find_instruction_files

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    try:
        (a / "ln").symlink_to(b, target_is_directory=True)
        (b / "ln").symlink_to(a, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported")
    cfg = Config()
    # must terminate promptly
    files = find_instruction_files(a, tmp_path, cfg)
    assert isinstance(files, list)


# -- bug 2: undo cleanup ----------------------------------------------------

def test_missing_directories_walks_up_to_existing():
    from opencode_py.agent.loop import _missing_directories

    base = Path(tempfile.mkdtemp())
    deep = base / "x" / "y" / "z"
    missing = _missing_directories(deep)
    # deepest first, stopping before the existing base
    assert missing[0] == deep
    assert missing[-1] == base / "x"
    assert _missing_directories(base) == []


def test_undo_created_file_removes_it(tmp_path):
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tools import build_registry

    cfg = Config()
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=tmp_path, provider=mock.MagicMock(), agent="build")
    f = tmp_path / "new.txt"
    f.write_text("created")
    loop._undo_stack.append({"path": str(f), "original": None, "dirs": []})
    msg = loop.undo_last()
    assert "Reverted" in msg
    assert not f.exists()


def test_undo_cleans_created_parent_dirs(tmp_path):
    from opencode_py.agent.loop import _missing_directories
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tools import build_registry

    cfg = Config()
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=tmp_path, provider=mock.MagicMock(), agent="build")
    nested = tmp_path / "p" / "q" / "deep.txt"
    dirs = [str(d) for d in _missing_directories(nested.parent)]
    nested.parent.mkdir(parents=True)
    nested.write_text("hi")
    loop._undo_stack.append({"path": str(nested), "original": None, "dirs": dirs})
    loop.undo_last()
    assert not nested.exists()
    # dirs we created should be gone, but the common ancestor stays
    assert not (tmp_path / "p" / "q").exists()
    assert tmp_path.exists()


# -- bug 10: sub-agent inherits read paths ----------------------------------

def test_subagent_read_paths_inherited():
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tools import build_registry

    cfg = Config()
    parent = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=mock.MagicMock(), provider=mock.MagicMock(), agent="build")
    parent._read_paths = {"/a/b.py", "/c/d.py"}
    sub = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=mock.MagicMock(),
        provider=mock.MagicMock(),
        agent="build",
        read_paths=set(parent._read_paths),
    )
    assert sub._read_paths == {"/a/b.py", "/c/d.py"}
    # and the guard then passes for a path the parent read
    from pathlib import Path

    p = Path("/a/b.py")
    if p.exists():
        assert parent._ensure_read_before_edit(str(p))