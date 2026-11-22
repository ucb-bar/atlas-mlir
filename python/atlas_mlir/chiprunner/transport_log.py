# Bring-up tooling: logical payload counts and host wall times, not wire traffic.
from datetime import datetime, timezone
import json
import time


class TransferLog:
    def __init__(self, logs, session):
        self.logs, self.session = logs, session
        self.started = time.monotonic()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.events = []
        self.log = (logs/'run.log').open('w')

    def command(self, command, byte_count=None, output_path=None):
        operation = command.split()[0]
        started = time.monotonic()
        event = dict(operation=operation, command=command, bytes=byte_count,
                     started_utc=datetime.now(timezone.utc).isoformat(), success=False)
        self.log.write('> '+command+'\n'); self.log.flush()
        try:
            response = self.session.command(command)
            event['success'] = True
            self.log.write(response+'\n')
            return response
        except Exception as error:
            event['error'] = str(error)
            raise
        finally:
            event['wall_seconds'] = time.monotonic()-started
            if output_path is not None:
                event['observed_bytes'] = output_path.stat().st_size if output_path.exists() else 0
            event['bytes_per_second'] = (event.get('observed_bytes', byte_count)/event['wall_seconds']
                                        if byte_count is not None and event['success'] else None)
            self.events.append(event)
            with (self.logs/'transport.jsonl').open('a') as stream:
                stream.write(json.dumps(event, sort_keys=True)+'\n')
            self.log.write(json.dumps(event, sort_keys=True)+'\n'); self.log.flush()

    def finish(self, verdict, **details):
        totals = {}
        for event in self.events:
            if event['bytes'] is not None:
                row = totals.setdefault(event['operation'], dict(attempted_bytes=0, completed_bytes=0, wall_seconds=0))
                row['attempted_bytes'] += event['bytes']
                row['completed_bytes'] += event.get('observed_bytes', event['bytes']) if event['success'] else 0
                row['wall_seconds'] += event['wall_seconds']
        summary = dict(verdict=verdict, started_utc=self.started_utc,
                       wall_seconds=time.monotonic()-self.started, operations=self.events,
                       transfer_totals=totals, byte_semantics='requested logical payload bytes; observed_bytes records dump file size; ELF PT_LOAD file bytes exclude zero-fill and metadata; not USB/JTAG wire bytes',
                       **details)
        (self.logs/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
        self.log.close()
        return summary
