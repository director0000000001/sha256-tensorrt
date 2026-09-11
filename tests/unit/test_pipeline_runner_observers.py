"""The runner's frame observers (day 16): every published snapshot reaches
each observer once, on the publishing thread; one that raises is logged
and the loop goes on; a removed one hears nothing more."""

from __future__ import annotations

from sha256.pipeline.runner import PipelineRunner, PipelineSnapshot
from tests.unit.test_pipeline_runner import FakeController, wait_until


class TestObservers:
    def test_every_published_frame_reaches_the_observer(self):
        controller = FakeController()
        runner = PipelineRunner(controller)  # type: ignore[arg-type]
        seen: list[int] = []
        runner.add_observer(lambda snapshot: seen.append(snapshot.frame_id))
        for _ in range(3):
            frame = controller.process_once()
            assert frame is not None
            runner._publish(frame)
        assert seen == [1, 2, 3]

    def test_adding_twice_is_once_and_removing_stops_it(self):
        controller = FakeController()
        runner = PipelineRunner(controller)  # type: ignore[arg-type]
        seen: list[PipelineSnapshot] = []
        runner.add_observer(seen.append)
        runner.add_observer(seen.append)
        runner._publish(controller.process_once())
        assert len(seen) == 1
        runner.remove_observer(seen.append)
        runner.remove_observer(seen.append)  # gone already: no error
        runner._publish(controller.process_once())
        assert len(seen) == 1

    def test_a_raising_observer_does_not_stop_the_loop(self):
        controller = FakeController()
        runner = PipelineRunner(controller)  # type: ignore[arg-type]
        after: list[int] = []

        def boom(snapshot: PipelineSnapshot) -> None:
            raise RuntimeError("listener broke")

        runner.add_observer(boom)
        runner.add_observer(lambda s: after.append(s.frame_id))
        runner.start()
        try:
            assert wait_until(lambda: len(after) >= 5)
            assert runner.last_error is None and runner.is_running
        finally:
            runner.stop()
