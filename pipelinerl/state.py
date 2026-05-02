import logging
import threading
import time
from pathlib import Path

from pydantic import TypeAdapter

from pipelinerl.finetune_loop import (
    TRAINER_TOPIC,
    TrainerMessage,
    WeightUpdateSuccess,
    SamplesProcessed,
)
from pipelinerl.streams import SingleStreamSpec, read_stream

logger = logging.getLogger(__name__)


class TrainerState:
    def __init__(self, exp_path: Path):
        self.exp_path = exp_path
        self.propagated_weight_version: int | None = None
        self.samples_processed: int | None = None
        self.completed_steps: int | None = None
        self._version_to_completed_steps: dict[int, int] = {}

    def debug_mode_init(self):
        self.propagated_weight_version = 0
        self.samples_processed = 0
        self.completed_steps = 0
        self._version_to_completed_steps[0] = 0

    def start_listening(self):
        stream = SingleStreamSpec(exp_path=self.exp_path, topic=TRAINER_TOPIC)

        def listen():
            with read_stream(stream) as reader:
                for line in reader.read():
                    message = TypeAdapter(TrainerMessage).validate_python(line)
                    if isinstance(message, WeightUpdateSuccess):
                        self.propagated_weight_version = message.version
                        if message.completed_steps is not None:
                            self.completed_steps = message.completed_steps
                            self._version_to_completed_steps[message.version] = message.completed_steps
                    if isinstance(message, SamplesProcessed):
                        self.samples_processed = message.samples_processed
                        if message.completed_steps is not None:
                            self.completed_steps = message.completed_steps

        self._thread = threading.Thread(target=listen)
        self._thread.start()
    
    def wait_for_processed_samples(self):
        while self.samples_processed is None:
            logger.info("Waiting for the trainer to declare the number of processed samples")
            time.sleep(1)
        return self.samples_processed

    def wait_for_model_version(self):
        while self.propagated_weight_version is None:
            logger.info("Waiting for the trainer to declare the initial weight version")
            time.sleep(1)
        return self.propagated_weight_version

    def get_completed_steps_for_version(self, version: int | None) -> int | None:
        if version is None:
            return self.completed_steps
        if version in self._version_to_completed_steps:
            return self._version_to_completed_steps[version]
        return self.completed_steps
