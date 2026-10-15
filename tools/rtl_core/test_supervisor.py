import unittest
from pathlib import Path
from unittest import mock
import supervisor


class DeclaredResourcePolicy(unittest.TestCase):
    def test_wrong_affinity_refuses_before_launch(self):
        with mock.patch.object(supervisor.os,"sched_getaffinity",return_value={0}), \
                mock.patch.object(supervisor.subprocess,"Popen") as launch:
            with self.assertRaises(RuntimeError):
                supervisor.bounded([],Path("."),{},Path("unused.log"),None,
                                   affinity={3},rss_limit=2*1024**3,wall_seconds=600)
            launch.assert_not_called()

    def test_invalid_limits_refuse_before_launch(self):
        with mock.patch.object(supervisor.os,"sched_getaffinity",return_value={3}), \
                mock.patch.object(supervisor.subprocess,"Popen") as launch:
            for rss,wall in ((0,600),(1024,0),(-1,600),(1024,-1)):
                with self.subTest(rss=rss,wall=wall),self.assertRaises(ValueError):
                    supervisor.bounded([],Path("."),{},Path("unused.log"),None,
                                       affinity={3},rss_limit=rss,wall_seconds=wall)
            launch.assert_not_called()


if __name__=="__main__":unittest.main()
