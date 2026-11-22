"""Laptop setup and publish scripts over portable stand-ins: a local ssh, a stub OpenOCD that is never executed, and a
synthetic packaged kit. They need no network, board, or real OpenOCD, and they do not qualify any laptop."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
SETUP, PUBLISH = REPO / 'scripts/laptop-setup.sh', REPO / 'scripts/laptop-publish.sh'
BANNER = 'Open On-Chip Debugger 0.12.0-laptop-test (2026-01-01-00:00)'

# ssh stand-in: runs the remote command here, in FAKE_REMOTE_HOME with a clean environment, as sshd would.
# FAKE_SSH_DROP fails a matching command without running it; FAKE_SSH_FAIL runs it, then fails as a dropped link would.
SSH = r'''#!/bin/sh
while [ $# -gt 0 ]; do case $1 in -n) exec < /dev/null; shift ;; -*) shift ;; *) break ;; esac; done
shift
printf '%s\n' "$*" >> "$FAKE_REMOTE_HOME/../ssh.log"
matches() { [ -n "$1" ] && case $2 in *"$1"*) true ;; *) false ;; esac; }
matches "${FAKE_SSH_DROP:-}" "$*" && exit 255
cd "$FAKE_REMOTE_HOME" || exit 255
env -i HOME="$FAKE_REMOTE_HOME" PATH=/usr/bin:/bin FAKE_DEPLOY_RC="${FAKE_DEPLOY_RC:-0}" /bin/sh -c "$*"
rc=$?
matches "${FAKE_SSH_FAIL:-}" "$*" && exit 255
exit $rc
'''
# atlasc stand-in on the remote host: `atlasc deploy EXPORT FLAGS -o OUT` copies EXPORT to OUT and records its flags.
# Like deploy, it ends a passing deploy by printing OUT's physical path.
ATLASC = r'''#!/bin/sh
[ "$1" = deploy ] || exit 2
eval "out=\${$#}"
mkdir "$out" && cp -R "$2"/. "$out"/ && echo "$*" > "$out/args"
echo "deploying $2"
[ "$FAKE_DEPLOY_RC" != 0 ] || (cd "$out" && pwd -P)
exit "$FAKE_DEPLOY_RC"
'''
# A packaged OUT whose staged runner is the copied package's, so that package's dry run accepts it.
BUILD_OUT = '''
import json, sys
from pathlib import Path
from atlas_mlir.deployment_kit import owned_tools
from test_chiprunner_inputs import live_kit
out = Path(sys.argv[1])
(out / 'kit/bench').mkdir(parents=True)
for name, source in owned_tools().items():
    (out / 'kit/bench' / name).write_bytes(source.read_bytes())
live_kit(out / 'kit')
(out / 'deploy-status.json').write_text(json.dumps(dict(status='PACKAGED', transport='jtag')))
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


class LaptopScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = root = Path(os.path.realpath(cls.tmp.name))
        cls.bin = root / 'bin'
        cls.bin.mkdir()
        executable(cls.bin / 'ssh', SSH)
        executable(cls.bin / 'atlasc', ATLASC)
        executable(cls.bin / 'uname', '#!/bin/sh\necho Linux\n')  # the Linux path on any host
        executable(cls.bin / 'fuser', '#!/bin/sh\nexit 1\n')  # setup needs one on PATH; it never runs here
        # An install laid out as a source checkout: a copy of this package whose table also declares the stub OpenOCD.
        package = root / 'install/python'
        shutil.copytree(REPO / 'python/atlas_mlir', package / 'atlas_mlir', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(REPO / 'pyproject.toml', root / 'install/pyproject.toml')
        prefix = root / 'openocd'
        (prefix / 'bin').mkdir(parents=True)
        (prefix / 'lib').mkdir()
        cls.openocd = prefix / 'bin/openocd'
        executable(cls.openocd, "#!/bin/sh\necho '%s' >&2\n" % BANNER)
        (prefix / 'lib/libstub.so.1').write_bytes(b'stub library')
        (prefix / 'lib/libstub.so').symlink_to('libstub.so.1')
        table = package / 'atlas_mlir/chiprunner/openocd-builds.json'
        rows = json.loads(table.read_text()) + [dict(sha256=sha256(cls.openocd), banner=BANNER, package='test stub')]
        table.write_text(json.dumps(rows))
        cls.activate = root / 'activate'
        cls.activate.write_text('PATH=%s:$PATH PYTHONPATH=%s\nexport PATH PYTHONPATH\n' % (shlex.quote(str(cls.bin)), shlex.quote(str(package))))
        cls.out = root / 'out'
        subprocess.run([sys.executable, '-B', '-c', BUILD_OUT, str(cls.out)], check=True, cwd=root,
                       env=dict(os.environ, PYTHONPATH='%s:%s' % (package, REPO / 'test')))
        cls.bundle = root / 'bundle'
        cls.published = cls.publish(cls.openocd, cls.bundle)
        cls.export = root / 'export'
        (cls.export / 'fx').mkdir(parents=True)
        (cls.export / 'fx/graph.json').write_text('{}')
        cls.home = root / 'laptop'
        cls.setup_run = cls.setup(cls.bundle, cls.home)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def publish(cls, openocd, dest, install=None, **env):
        activate, out = (install['activate'], install['out']) if install else (cls.activate, cls.out)
        return subprocess.run(['bash', str(PUBLISH), str(activate), str(out), str(dest)], capture_output=True, text=True,
                              env=dict(os.environ, NOV_OPENOCD=str(openocd), PYTHON=sys.executable, **env), timeout=300)

    def read_only_install(self):
        """A copy of the stand-in install (with bytecode in its package), its packaged OUT, and its OpenOCD prefix, all
        read-only as a real install is, and an empty TMPDIR for the publish."""
        base = Path(tempfile.mkdtemp(dir=self.root))
        shutil.copytree(self.root / 'install', base / 'install')
        cache = base / 'install/python/atlas_mlir/__pycache__'
        cache.mkdir()
        (cache / 'cli.cpython-test.pyc').write_bytes(b'stale bytecode')
        shutil.copytree(self.out, base / 'out')
        shutil.copytree(self.openocd.parent.parent, base / 'openocd', symlinks=True)
        activate = self.activate.read_text()
        self.assertIn(str(self.root / 'install/python'), activate)
        (base / 'activate').write_text(activate.replace(str(self.root / 'install/python'), str(base / 'install/python')))
        (base / 'tmp').mkdir()
        self.addCleanup(subprocess.run, ['chmod', '-R', 'u+w', str(base)])
        subprocess.run(['chmod', '-R', 'a-w', str(base / 'install'), str(base / 'out'), str(base / 'openocd')], check=True)
        return dict(activate=base / 'activate', out=base / 'out', openocd=base / 'openocd/bin/openocd', tmp=base / 'tmp', base=base)

    def assertPublishedNothing(self, dest, install):
        self.assertFalse(dest.exists())
        self.assertFalse(Path(str(dest) + '.partial').exists())
        self.assertEqual(list(install['tmp'].iterdir()), [])

    @classmethod
    def laptop(cls, command, home, **env):
        """A command on a clean laptop: only HOME, PATH, the Python, and the stand-ins' settings."""
        remote = Path(tempfile.mkdtemp(dir=cls.root)) / 'home'
        remote.mkdir()
        environment = dict(dict(HOME=str(home), PATH='%s:/usr/bin:/bin' % cls.bin, ATLASC_PYTHON=sys.executable,
                                FAKE_REMOTE_HOME=str(remote)), **env)
        result = subprocess.run(command, capture_output=True, text=True, env=environment, cwd=cls.root, timeout=300)
        result.remote = remote
        return result

    @classmethod
    def setup(cls, bundle, home, **env):
        home.mkdir()
        return cls.laptop(['sh', str(SETUP), 'bwrc:%s' % bundle], home, **env)

    def deploy(self, out, **env):
        return self.laptop([str(self.home / '.atlasc/bin/atlasc'), 'deploy', str(self.export), '--jtag', '-o', str(out)], self.home, **env)

    def variant(self, change):
        """A copy of the published bundle, changed, with LINKS relisted so that only the change differs."""
        bundle = Path(tempfile.mkdtemp(dir=self.root)) / 'bundle'
        shutil.copytree(self.bundle, bundle, symlinks=True)
        change(bundle)
        sums = bundle / 'SHA256SUMS'
        lines = [line if not line.endswith('  LINKS\n') else '%s  LINKS\n' % sha256(bundle / 'LINKS') for line in sums.read_text().splitlines(True)]
        if sums.stat().st_size:
            sums.write_text(''.join(lines))
        return bundle

    def test_publish_lists_files_and_link_targets(self):
        self.assertEqual(self.published.returncode, 0, self.published.stdout + self.published.stderr)
        self.assertIn('RUN DRY PASS', self.published.stdout)
        self.assertEqual((self.bundle / 'LINKS').read_text(), 'openocd/linux-64/lib/libstub.so -> libstub.so.1\n')
        sums = (self.bundle / 'SHA256SUMS').read_text()
        self.assertIn('%s  openocd/linux-64/bin/openocd\n' % sha256(self.openocd), sums)
        self.assertIn('%s  LINKS\n' % sha256(self.bundle / 'LINKS'), sums)
        self.assertNotIn('libstub.so\n', sums)

    def test_publish_refuses_an_openocd_outside_a_bin_directory(self):
        other = self.root / 'misplaced/openocd'
        other.parent.mkdir()
        shutil.copy2(self.openocd, other)  # declared, but not PREFIX/bin/openocd
        result = self.publish(other, self.root / 'misplaced-bundle')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn('NOV_OPENOCD must be PREFIX/bin/openocd', result.stderr)
        self.assertFalse((self.root / 'misplaced-bundle').exists())

    def test_publish_refuses_a_link_that_does_not_resolve_inside_its_part(self):
        for label, links, message in [
                ('escaping', [('libescape.so', '../../../reference')], 'libescape.so -> ../../../reference'),
                ('chain', [('a', '.'), ('b', 'a/..'), ('c', 'b/..'), ('d', 'c/..')], 'd -> c/..'),
                ('cycle', [('loop1', 'loop2'), ('loop2', 'loop1')], 'loop1 -> loop2'),
                ('dangling', [('libgone.so', 'libgone.so.1')], 'libgone.so -> libgone.so.1')]:
            with self.subTest(label):
                prefix = self.root / ('unresolved-' + label)
                shutil.copytree(self.openocd.parent.parent, prefix, symlinks=True)
                for name, target in links:
                    (prefix / 'lib' / name).symlink_to(target)
                result = self.publish(prefix / 'bin/openocd', self.root / ('unresolved-bundle-' + label))
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('link does not resolve inside its part: openocd/linux-64/lib/' + message, result.stderr)
                self.assertFalse((self.root / ('unresolved-bundle-' + label)).exists())

    def test_publish_from_a_read_only_install_gives_a_bundle_laptops_can_replace(self):
        install = self.read_only_install()
        dest = install['base'] / 'bundle'
        result = self.publish(install['openocd'], dest, install, TMPDIR=str(install['tmp']))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('RUN DRY PASS', result.stdout)
        self.assertFalse(Path(str(dest) + '.partial').exists())
        self.assertEqual(list(install['tmp'].iterdir()), [])
        self.assertEqual([path for path in dest.rglob('*') if path.name == '__pycache__'], [])
        home = install['base'] / 'laptop'
        home.mkdir()
        for attempt in 'first', 'again':  # a rerun replaces the laptop's copy of the bundle
            with self.subTest('setup ' + attempt):
                result = self.laptop(['sh', str(SETUP), 'bwrc:%s' % dest], home)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('RUN DRY PASS', result.stdout)
        self.assertEqual([path for path in [dest, *dest.rglob('*')] if not path.is_symlink() and not path.stat().st_mode & 0o200], [])

    def test_publish_from_a_read_only_install_reruns_after_a_failed_or_interrupted_publish(self):
        install = self.read_only_install()
        undeclared = install['base'] / 'undeclared/bin/openocd'
        undeclared.parent.mkdir(parents=True)
        undeclared.write_text('#!/bin/sh\n')
        mark = install['base'] / 'paused'
        # The install's Python, paused where the publish checks NOV_OPENOCD (after it has copied the read-only package and kit).
        paused = install['base'] / 'paused-python'
        executable(paused, '#!/bin/sh\n[ "$1 $2" = "-B -" ] && { : > %s; sleep 300; }\nexec %s "$@"\n' % (shlex.quote(str(mark)), shlex.quote(sys.executable)))
        for label in 'failed', 'SIGINT', 'SIGTERM', 'SIGHUP', 'SIGKILL':
            with self.subTest(label):
                dest = install['base'] / ('bundle-' + label)
                env = dict(TMPDIR=str(install['tmp']))
                if label == 'failed':
                    result = self.publish(undeclared, dest, install, **env)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('NOV_OPENOCD is not a declared OpenOCD build', result.stderr)
                else:
                    mark.unlink(missing_ok=True)
                    process = subprocess.Popen(['bash', str(PUBLISH), str(install['activate']), str(install['out']), str(dest)],
                                               env=dict(os.environ, NOV_OPENOCD=str(install['openocd']), PYTHON=str(paused), **env),
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                    deadline = time.monotonic() + 120
                    while not mark.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertTrue(mark.exists(), 'the publish never reached its NOV_OPENOCD check')
                    os.killpg(process.pid, getattr(signal, label))
                    output = process.communicate(timeout=60)[0]
                    self.assertNotEqual(process.returncode, 0, output)
                if label == 'SIGKILL':  # nothing can clean up: a rerun refuses and says how to remove what is left
                    result = self.publish(install['openocd'], dest, install, **env)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn('%s.partial is left from a killed or running publish' % dest, result.stderr)
                    subprocess.run(['sh', '-c', 'chmod -R u+w "$1.partial"; rm -rf "$1.partial"', '_', str(dest)], check=True)
                    for leftover in install['tmp'].iterdir():
                        subprocess.run(['sh', '-c', 'chmod -R u+w "$1"; rm -rf "$1"', '_', str(leftover)], check=True)
                self.assertPublishedNothing(dest, install)
                result = self.publish(install['openocd'], dest, install, **env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('RUN DRY PASS', result.stdout)
                self.assertEqual(list(install['tmp'].iterdir()), [])

    def test_setup_installs_and_dry_runs_the_reference_kit(self):
        result = self.setup_run
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('RUN DRY PASS', result.stdout)
        self.assertNotIn('not declared', result.stdout)
        env = (self.home / '.atlasc/env').read_text()
        self.assertIn("NOV_OPENOCD='%s'" % (self.home / '.atlasc/bundle/openocd/linux-64/bin/openocd'), env)
        self.assertIn("ATLASC_ACTIVATE='%s'" % self.activate, env)
        self.assertEqual(os.readlink(self.home / '.atlasc/bundle/openocd/linux-64/lib/libstub.so'), 'libstub.so.1')

    def test_setup_refuses_a_bundle_its_listings_do_not_match(self):
        def write(name, text):
            return lambda bundle: (bundle / name).write_text(text)

        def relink(target, name='libstub.so'):
            def change(bundle):
                (bundle / 'openocd/linux-64/lib' / name).unlink(missing_ok=True)
                (bundle / 'openocd/linux-64/lib' / name).symlink_to(target)
                if name == 'libstub.so':
                    (bundle / 'LINKS').write_text('openocd/linux-64/lib/libstub.so -> %s\n' % target)
            return change
        def add_links(*pairs):
            def change(bundle):
                for name, target in pairs:
                    (bundle / name).symlink_to(target)
                with open(bundle / 'LINKS', 'a') as stream:
                    stream.writelines('%s -> %s\n' % pair for pair in pairs)
            return change
        library = str(self.bundle / 'openocd/linux-64/lib/libstub.so.1')
        lib = 'openocd/linux-64/lib/'
        for label, change, message in [
                ('an unlisted file', write('python/sitecustomize.py', 'import os\n'), 'unlisted file: python/sitecustomize.py'),
                ('a changed file', write('python/atlas_mlir/cli.py', '#'), 'differs: python/atlas_mlir/cli.py'),
                ('an empty SHA256SUMS', write('SHA256SUMS', ''), 'nothing listed for'),
                ('an unlisted link', relink('libstub.so.1', 'libextra.so'), 'unlisted link: openocd/linux-64/lib/libextra.so'),
                ('a listed absolute link', relink(library), 'link does not resolve inside its part: openocd/linux-64/lib/libstub.so'),
                ('a listed escaping link', relink('../../../python/atlas_mlir/cli.py'), 'link does not resolve inside its part: openocd/linux-64/lib/libstub.so'),
                ('a chain of links that leaves its part', add_links(('python/a', '.'), ('python/b', 'a/..'), ('python/c', 'b/..')),
                 'link does not resolve inside its part: python/b'),
                ('a chain whose last link leaves its part',
                 add_links((lib + 'a', '.'), (lib + 'b', 'a/..'), (lib + 'c', 'b/..'), (lib + 'd', 'c/..')),
                 'link does not resolve inside its part: openocd/linux-64/lib/d'),
                ('a link that leaves and reenters its part', add_links((lib + 'back', '../../../python/../openocd/linux-64/lib/libstub.so.1')),
                 'link does not resolve inside its part: openocd/linux-64/lib/back'),
                ('a link cycle', add_links((lib + 'loop1', 'loop2'), (lib + 'loop2', 'loop1')), 'link does not resolve inside its part: openocd/linux-64/lib/loop1'),
                ('a dangling link', add_links((lib + 'libgone.so', 'libgone.so.1')), 'link does not resolve inside its part: openocd/linux-64/lib/libgone.so'),
                ('a missing file', lambda bundle: (bundle / 'python/atlas_mlir/cli.py').unlink(), 'missing: python/atlas_mlir/cli.py')]:
            with self.subTest(label):
                home = Path(tempfile.mkdtemp(dir=self.root)) / 'home'
                result = self.setup(self.variant(change), home)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse((home / '.atlasc/bin/atlasc').exists())

    def test_linux_setup_refuses_without_pgrep_or_fuser(self):
        for missing in ('pgrep', 'fuser'):
            with self.subTest(missing):
                tools = Path(tempfile.mkdtemp(dir=self.root))
                for name in ('dirname', 'mkdir', 'grep', 'pgrep', 'fuser'):
                    if name != missing:
                        executable(tools / name, '#!/bin/sh\nexec %s "$@"\n' % (shutil.which(name, path='/usr/bin:/bin') or self.bin / name))
                shutil.copy2(self.bin / 'uname', tools / 'uname')
                home = Path(tempfile.mkdtemp(dir=self.root)) / 'home'
                home.mkdir()
                result = self.laptop(['/bin/sh', str(SETUP), 'bwrc:%s' % self.bundle], home, PATH=str(tools))
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('laptop-setup: %s not found; live runs need pgrep and fuser: install procps and psmisc' % missing, result.stderr)
                self.assertFalse((home / '.atlasc/bundle').exists())

    def test_setup_refuses_when_ssh_fails_after_a_valid_archive(self):
        self.assertEqual(self.published.returncode, 0, 'needs the published bundle')
        home = self.root / 'dropped-fetch'
        result = self.setup(self.bundle, home, FAKE_SSH_FAIL='tar -cf -')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('could not fetch', result.stderr)
        self.assertFalse((home / '.atlasc/bin/atlasc').exists())

    def test_macos_setup_keeps_only_a_declared_openocd_and_checks_micromamba(self):
        """A stand-in Mac (uname, curl, and shasum): an installed OpenOCD is kept only when its digest is declared, and a
        micromamba download that differs from its pin refuses before anything runs."""
        mac = self.root / 'mac-bin'
        mac.mkdir(exist_ok=True)
        executable(mac / 'uname', '#!/bin/sh\ncase $1 in -m) echo x86_64 ;; *) echo Darwin ;; esac\n')
        executable(mac / 'curl', '#!/bin/sh\necho "$*" >> "$HOME/curl.log"\nwhile [ "$1" != -o ]; do shift; done\necho not-micromamba > "$2"\n')
        executable(mac / 'shasum', '#!/bin/sh\nshift 2\nexec %s -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], \'rb\').read()).hexdigest(), sys.argv[1])" "$1"\n'
                   % shlex.quote(sys.executable))
        for label, installed, downloads in [('fresh', None, True), ('partial', b'#!/bin/sh\n', True), ('declared', self.openocd.read_bytes(), False)]:
            with self.subTest(label):
                home = Path(tempfile.mkdtemp(dir=self.root)) / 'home'
                if installed is not None:
                    binary = home / '.atlasc/ocd/bin/openocd'
                    binary.parent.mkdir(parents=True)
                    binary.write_bytes(installed)
                    binary.chmod(0o755)
                home.mkdir(exist_ok=True)
                result = self.laptop(['sh', str(SETUP), 'bwrc:%s' % self.bundle], home, PATH='%s:%s:/usr/bin:/bin' % (mac, self.bin))
                self.assertEqual((home / 'curl.log').exists(), downloads, result.stdout + result.stderr)
                if downloads:
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn('micromamba differs from its pinned sha256', result.stderr)
                    self.assertFalse((home / '.atlasc/bin/atlasc').exists())
                    self.assertFalse((home / '.atlasc/ocd').exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('RUN DRY PASS', result.stdout)
                    self.assertIn("NOV_OPENOCD='%s'" % (home / '.atlasc/ocd/bin/openocd'), (home / '.atlasc/env').read_text())

    def test_macos_setup_installs_a_declared_build(self):
        """The conda-forge package macOS setup installs is the one the runner's table declares for macOS."""
        rows = json.loads((REPO / 'python/atlas_mlir/chiprunner/openocd-builds.json').read_text())
        self.assertIn(" --platform osx-64 \\\n      --override-channels -c conda-forge 'openocd=0.12.0=h57f6e69_0' ", SETUP.read_text())
        self.assertIn('conda-forge osx-64 openocd-0.12.0-h57f6e69_0', [row['package'] for row in rows])

    def test_failed_cleanup_exits_nonzero_and_keeps_both_copies_of_out(self):
        for label, env, rc in [('deploy passed', {}, 1), ('deploy failed', dict(FAKE_DEPLOY_RC='3'), 3)]:
            with self.subTest(label):
                out = self.root / ('uncleaned-' + label.split()[1])
                result = self.deploy(out, FAKE_SSH_DROP='rm -rf', **env)
                self.assertEqual(result.returncode, rc, result.stdout + result.stderr)
                [work] = self.remote_work(result)
                self.assertIn('atlasc deploy: cleanup failed; the remote files stay in bwrc:atlasc-remote/%s\n' % work, result.stderr)
                self.assertTrue((out / 'args').exists())
                self.assertTrue((result.remote / 'atlasc-remote' / work / 'out/args').exists())

    def test_macos_deploy_archives_no_extended_attributes(self):
        """A stand-in macOS tar: unless given --no-xattrs it adds the extended attribute header that the Mac's tar wrote
        for each file, which GNU tar on BWRC reported as an unknown keyword."""
        mac = Path(tempfile.mkdtemp(dir=self.root))
        executable(mac / 'uname', '#!/bin/sh\necho Darwin\n')
        executable(mac / 'tar', '#!/bin/sh\ncase " $* " in *" -cf "*) case " $* " in *" --no-xattrs "*) ;; *) set -- --format=pax '
                   '--pax-option=LIBARCHIVE.xattr.com.apple.provenance:=AQ "$@" ;; esac ;; esac\nexec /usr/bin/env PATH=/usr/bin:/bin tar "$@"\n')
        out = self.root / 'deployed-from-mac'
        result = self.deploy(out, PATH='%s:%s:/usr/bin:/bin' % (mac, self.bin))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('extended header', result.stderr)
        self.assertEqual((out / 'fx/graph.json').read_text(), '{}')

    def remote_work(self, result):
        return sorted(path.name for path in (result.remote / 'atlasc-remote').iterdir())

    def test_deploy_runs_remotely_and_copies_out_back(self):
        out = self.root / 'deployed'
        result = self.deploy(out)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((out / 'args').read_text(), 'deploy model --jtag -o out\n')
        self.assertEqual((out / 'fx/graph.json').read_text(), '{}')
        self.assertEqual(self.remote_work(result), [])
        self.assertEqual(result.stdout, 'deploying model\n%s\n' % out)  # the local OUT, not the remote one

    def test_failed_deploy_still_returns_its_out(self):
        out = self.root / 'refused'
        result = self.deploy(out, FAKE_DEPLOY_RC='3')
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(result.stdout, 'deploying model\n')
        self.assertTrue((out / 'args').exists())
        self.assertEqual(self.remote_work(result), [])

    def test_failed_probe_or_retrieval_exits_nonzero_and_keeps_the_remote_out(self):
        for label, env in [('probe', dict(FAKE_SSH_DROP='test -d')), ('retrieval', dict(FAKE_SSH_FAIL='/out -cf -'))]:
            with self.subTest(label):
                out = self.root / ('lost-' + label)
                result = self.deploy(out, **env)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('the remote files stay in bwrc:atlasc-remote/', result.stderr)
                [work] = self.remote_work(result)
                self.assertTrue((result.remote / 'atlasc-remote' / work / 'out/args').exists())

    def test_failed_upload_exits_before_the_remote_deploy(self):
        unreadable = self.root / 'unreadable-export'
        shutil.copytree(self.export, unreadable)
        (unreadable / 'fx/graph.json').chmod(0)
        self.addCleanup((unreadable / 'fx/graph.json').chmod, 0o644)
        for label, source, env in [('archive', unreadable, {}), ('ssh', self.export, dict(FAKE_SSH_FAIL='/model -xf -'))]:
            with self.subTest(label):
                out = self.root / ('upload-' + label)
                result = self.laptop([str(self.home / '.atlasc/bin/atlasc'), 'deploy', str(source), '--jtag', '-o', str(out)], self.home, **env)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertFalse(out.exists())
                self.assertNotIn('atlasc deploy', (result.remote.parent / 'ssh.log').read_text() if (result.remote.parent / 'ssh.log').exists() else '')


if __name__ == '__main__':
    unittest.main()
