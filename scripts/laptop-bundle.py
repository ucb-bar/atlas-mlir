"""List or check a laptop run bundle (docs/laptop.md); laptop-publish.sh and laptop-setup.sh run it.
usage: laptop-bundle.py list ROOT          write LINKS and SHA256SUMS for every link and file under ROOT
       laptop-bundle.py check ROOT PART... refuse ROOT unless its entries are exactly those its listings name
Every link must resolve, through the links it reaches, to an existing entry without leaving its top-level part."""
import hashlib
import os
import re
import sys


def resolve(root, rel):
    """The path that rel's link chain reaches under root, or None if a step leaves rel's part, loops, or finds nothing."""
    part, *pending = rel.split('/')
    if os.path.islink(os.path.join(root, part)):
        return None
    done, hops = [part], 0
    while pending:
        name = pending.pop(0)
        if name in ('', '.'):
            continue
        if name == '..':
            if len(done) == 1:
                return None
            done.pop()
            continue
        path = os.path.join(root, *done, name)
        if os.path.islink(path):
            hops += 1
            target = os.readlink(path)
            if hops > 40 or target.startswith('/'):
                return None
            pending[:0] = target.split('/')
        elif os.path.isdir(path) or (os.path.exists(path) and not pending):
            done.append(name)
        else:
            return None
    return '/'.join(done)


def entries(root):
    for top, dirs, names in os.walk(root):
        for name in sorted(dirs + names):
            path = os.path.join(top, name)
            yield path, os.path.relpath(path, root)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def write_listings(root):
    files, links = [], []
    for path, rel in entries(root):
        if os.path.islink(path):
            if resolve(root, rel) is None:
                sys.exit('laptop-publish: link does not resolve inside its part: %s -> %s' % (rel, os.readlink(path)))
            links.append('%s -> %s\n' % (rel, os.readlink(path)))
        elif os.path.isfile(path):
            files.append(rel)
    with open(os.path.join(root, 'LINKS'), 'w') as stream:
        stream.writelines(sorted(links))
    with open(os.path.join(root, 'SHA256SUMS'), 'w') as stream:
        stream.writelines('%s  %s\n' % (sha256(os.path.join(root, name)), name) for name in sorted(files + ['LINKS']))


def listing(root, name, pattern):
    path = os.path.join(root, name)
    if os.path.islink(path) or not os.path.isfile(path):
        sys.exit('malformed ' + name)
    with open(path) as stream:
        rows = [re.fullmatch(pattern, line) for line in stream]
    if not all(rows):
        sys.exit('malformed ' + name)
    return [row.groups() for row in rows]


def check(root, parts):
    files = {name: digest for digest, name in listing(root, 'SHA256SUMS', r'([0-9a-f]{64})  (.+)\n')}
    links = dict(listing(root, 'LINKS', r'(.+) -> (.+)\n'))
    expected = {name for name in [*files, *links] if name.split('/')[0] in parts}
    for part in parts - {'SHA256SUMS'}:
        if not any(name.split('/')[0] == part for name in expected):
            sys.exit('nothing listed for ' + part)
    found = set()
    for path, rel in entries(root):
        if rel.split(os.sep)[0] not in parts:
            sys.exit('unexpected ' + rel)
        if os.path.islink(path):
            if links.get(rel) != os.readlink(path):
                sys.exit('unlisted link: ' + rel)
            if resolve(root, rel) is None:
                sys.exit('link does not resolve inside its part: ' + rel)
        elif os.path.isfile(path):
            if rel == 'SHA256SUMS':
                continue
            if rel not in files:
                sys.exit('unlisted file: ' + rel)
            if sha256(path) != files[rel]:
                sys.exit('differs: ' + rel)
        elif os.path.isdir(path):
            continue
        else:
            sys.exit('not a file, directory, or link: ' + rel)
        found.add(rel)
    if expected - found:
        sys.exit('missing: ' + ', '.join(sorted(expected - found)))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'list':
        write_listings(sys.argv[2])
    elif len(sys.argv) > 3 and sys.argv[1] == 'check':
        check(sys.argv[2], set(sys.argv[3:]))
    else:
        sys.exit('usage: laptop-bundle.py list ROOT | laptop-bundle.py check ROOT PART...')
