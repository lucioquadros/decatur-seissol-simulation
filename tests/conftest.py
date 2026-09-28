from pathlib import Path

import pytest
import yaml

SCENARIOS = ("bob_will", "nick")


class EasiLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        out = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            out[tuple(key) if isinstance(key, list) else key] = \
                self.construct_object(value_node, deep=True)
        return out


def _tagged(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node, deep=True)
    else:
        value = loader.construct_scalar(node)
    return {"tag": tag_suffix, "value": value}


EasiLoader.add_multi_constructor("!", _tagged)


def load_easi(text: str) -> dict:
    """easi YAML with custom tags as {'tag', 'value'}, flow-sequence keys as tuples."""
    return yaml.load(text, Loader=EasiLoader)


def find_tagged(node, tag: str) -> list[dict]:
    """Values of every component with this tag, at any depth."""
    found = []
    if isinstance(node, dict):
        if node.get("tag") == tag:
            found.append(node["value"])
        for v in node.values():
            found += find_tagged(v, tag)
    elif isinstance(node, list):
        for v in node:
            found += find_tagged(v, tag)
    return found


def lua_functions(text: str) -> dict:
    """Returned names (tuple) -> Lua source of each !LuaMap in an easi file."""
    return {tuple(m["returns"]): m["function"] for m in find_tagged(load_easi(text), "LuaMap")}


@pytest.fixture
def lua():
    lupa = pytest.importorskip("lupa")
    runtime = lupa.LuaRuntime()
    compiled = {}

    def call(source: str, *xyz: float, **inputs: float) -> dict:
        if source not in compiled:
            compiled[source] = runtime.execute(source + "\nreturn f")
        f = compiled[source]
        out = f(runtime.table_from(dict(zip("xyz", xyz)) | inputs))
        return {k: out[k] for k in out.keys()}

    return call


def data_dir() -> Path | None:
    from decatur.config import load_paths
    d = load_paths().get("DATA_DIR")
    return d if d and d.is_dir() else None
