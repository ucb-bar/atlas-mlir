"""Persistent Tcl adapter. Session ownership and live preflight belong to caller."""
import json
import time
if __package__:
    from .kit_manifest import safe_path, sha
else:
    from kit_manifest import safe_path, sha
if __package__:
    from .tcl_rpc import TransportError, tcl_path
else:
    from tcl_rpc import TransportError, tcl_path
if __package__:
    from .composed_sim import _payload_elf, runner_ranges, step_loads
else:
    from composed_sim import _payload_elf, runner_ranges, step_loads


class TclTransport:
    """One OpenOCD Tcl session. With `bench` (the runner's bench arguments) it owns
    the OpenOCD process and the UART that linked step ELFs report on."""

    def __init__(self, root, work, manifest, *, rpc=None, bench=None, **unused):
        self.root, self.work, self.rpc = root, work, rpc
        self.tag = manifest.get('runner', {}).get('record_tag')
        self.sequence = 0
        self.process = self.uart = self.openocd_log = None
        self.verified = {}
        if bench is not None:
            if __package__:
                from .preflight import bench as bench_preflight
                from .run_kit import UART, connect, launch_openocd
            else:
                from preflight import bench as bench_preflight
                from run_kit import UART, connect, launch_openocd
            # The single-kit rule: a declared OpenOCD build, no other bench process, a free UART.
            baud = manifest['composed'].get('receiver_baud')
            if baud is None:
                raise ValueError('live step ELFs need the plan receiver_baud')
            bench_preflight(bench.openocd, bench.serial)
        # Startup cleanup covers the initial reset: any failure closes what this transport opened, without retry.
        try:
            if bench is not None:
                self.uart = UART(bench.serial, baud)
                self.openocd_log = (work/'openocd.log').open('w')
                self.process = launch_openocd(bench, safe_path(root/'bench'/'atlas-soc.cfg'), work, self.openocd_log)
                self.rpc = connect(bench, self.process)
            self.rpc.command('reset halt')
        except BaseException:
            self._release(); raise

    def write(self, address, data):
        """Load `data` and return the bytes then read back, which must equal it byte for byte."""
        self.verified = {}
        path=safe_path(self.work/f'transfer-{self.sequence}.bin'); self.sequence+=1
        path.write_bytes(data)
        self.rpc.command(f'load_image {tcl_path(path)} {address:#x} bin')
        # The same JTAG reads verify_image makes with no work area (atlas-soc.cfg), which compares only a CRC-32.
        readback, observed = self._dump(address, len(data))
        if observed!=data: raise TransportError(f'write readback differs at {address:#x}')
        readback.unlink()  # kept only when it differs
        return observed

    def read(self, address, size):
        if (address, size) in self.verified:
            # The bytes the step's UART hash was checked against, not a second dump.
            return self.verified[address, size]
        return self._dump(address, size)[1]

    def _dump(self, address, size):
        path=safe_path(self.work/f'transfer-{self.sequence}.bin'); self.sequence+=1
        path.unlink(missing_ok=True)
        self.rpc.command(f'dump_image {tcl_path(path)} {address:#x} {size}')
        data=path.read_bytes()
        if len(data)!=size: raise TransportError('short Tcl output dump')
        return path, data

    def prepare(self, region):
        # Device harnesses, host operations, and step ELFs (host code and window image) are linked ELFs.
        self.verified = {}
        elf=safe_path(self.root/region['elf'])
        if region['kind']=='elf':
            # A step ELF loads without the bytes it clears at startup and the ranges the runner overwrites next.
            # Interim: a sparse ELF with NOBITS zero spans would make the cleared-span part loader-independent.
            raw=elf.read_bytes()
            pieces,left_out=step_loads(raw, runner_ranges(region))
            if left_out:
                elf=safe_path(self.work/f'image-{self.sequence}.elf'); self.sequence+=1
                elf.write_bytes(_payload_elf(raw[:64], pieces))
        self.rpc.command('load_image '+tcl_path(elf))
        self.rpc.command('verify_image '+tcl_path(elf))

    def launch(self, region):
        self.verified = {}
        if region['kind']=='elf':
            if self.uart is None: raise TransportError('step ELFs report on the UART, which is not open')
            self.uart.clear()
            self.rpc.command(f'resume {region["entry"]:#x}')
            return
        status=region['status']
        initial=status.get('initial_value',0)
        if initial==status['done_value']: raise ValueError('completion sentinel must change')
        self.write(status['address'],initial.to_bytes(status.get('bytes',8),'little'))
        self.rpc.command(f'resume {region["entry"]:#x}')

    def wait(self, region, timeout):
        if region['kind']=='elf':
            return self._step(region, timeout)
        deadline=time.monotonic()+timeout
        status=region['status']
        while time.monotonic()<deadline:
            original_timeout=self.rpc.timeout
            try:
                self.rpc.timeout=min(original_timeout,max(.001,deadline-time.monotonic()))
                value=int.from_bytes(self.read(status['address'],status.get('bytes',8)),'little')
            finally:
                self.rpc.timeout=original_timeout
            if value==status['done_value']:
                self.rpc.command('halt'); return
            if value!=status.get('initial_value',0):
                raise TransportError('unexpected completion status')
            time.sleep(min(.05,max(0,deadline-time.monotonic())))
        raise TransportError('region completion timeout')

    def _step(self, region, timeout):
        """UART completion, then every declared hash range dumped and equal to its UART hash."""
        if __package__:
            from .run_kit import step_hashes
        else:
            from run_kit import step_hashes
        log=safe_path(self.work/f'uart-{self.sequence}-{region["id"]}.log'); self.sequence+=1
        with log.open('wb') as console:
            records=self.uart.records(timeout, console, self.tag)
        self.rpc.command('halt')
        rows=json.loads(safe_path(self.root/region['deployment']).read_text())['hash_ranges']
        hashes=step_hashes(rows, records.decode('utf-8', 'replace'), self.tag)
        verified={}
        for row in rows:
            data=self.read(row['address'],row['bytes'])
            if sha(data)!=hashes[row['id']]: raise TransportError('dump differs from execution hash: '+row['id'])
            verified[row['address'],row['bytes']]=data
        self.verified=verified

    def close(self):
        # Never reset/retry a failed region. The owner closes its single session.
        try:
            if self.rpc is not None and not self.rpc.broken: self.rpc.command('halt')
        finally:
            self._release()

    def _release(self):
        """Close the Tcl connection, stop OpenOCD, and close the UART and log. Every step runs even if an earlier one
        raises; the first error is raised after the last step."""
        process, steps = self.process, []
        if process is not None:
            def stop():
                process.terminate()
                try: process.wait(timeout=5)
                except Exception: process.kill(); process.wait(timeout=5)
            steps += [self.rpc.close] if self.rpc is not None else []
            steps.append(stop)
        steps += [item.close for item in (self.uart, self.openocd_log) if item is not None]
        self.process = self.uart = self.openocd_log = None
        first = None
        for step in steps:
            try: step()
            except BaseException as error: first = first or error
        if first is not None: raise first
