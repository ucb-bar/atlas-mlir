"""Admit physical paths without parent traversal or symbolic links."""

from pathlib import Path
import stat


def lexical_path(value, *, absolute=False):
    text = str(value)
    path = Path(text)
    if (absolute and not path.is_absolute() or ".." in path.parts or "\\" in text
        or any(ord(c) < 32 or ord(c) == 127 for c in text)):
        raise ValueError("invalid lexical path")
    return path


def checked(value):
    path = lexical_path(value).absolute()
    lexical_path(path, absolute=True)
    for prefix in (*reversed(path.parents), path):
        try:
            mode = prefix.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError("provide physical paths without symbolic-link components")
    return path


def lexical_absolute_path(value):
    """Validate metadata syntax without accessing its target."""
    return lexical_path(value, absolute=True)


def map_metadata_paths(value, mapper, *, mapped_paths=None):
    """Return a relocated JSON view; callers must bind the original bytes first.

    Only absolute string values and absolute dictionary keys are mapped. A shared
    cache rejects aliases across related documents. The mapper must be lexical;
    physical admission still occurs at each existing file-read boundary.
    """
    if mapper is None:
        return value
    paths = []

    def collect(item):
        if isinstance(item, str) and item.startswith("/"):
            lexical_absolute_path(item)
            paths.append(item)
        elif isinstance(item, list):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                collect(key)
                collect(child)

    collect(value)  # Screen every original path before invoking the callback.
    cache = {} if mapped_paths is None else dict(mapped_paths)
    reverse = {new: old for old, new in cache.items()}
    for old in paths:
        if old in cache:
            continue
        new = str(lexical_absolute_path(mapper(Path(old))))
        if new in reverse and reverse[new] != old:
            raise ValueError("Metadata path mapping collision")
        cache[old], reverse[new] = new, old

    def rewrite(item):
        if isinstance(item, str):
            return cache.get(item, item)
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                mapped_key = rewrite(key)
                if mapped_key in result:
                    raise ValueError("Metadata key mapping collision")
                result[mapped_key] = rewrite(child)
            return result
        return item

    result = rewrite(value)
    if mapped_paths is not None:
        mapped_paths.update(cache)
    return result
