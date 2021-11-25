from enum import Enum


class StepIntervalUnit(Enum):
    """
    By which interval we perform the step for learning rate schedulers.
    Attributes:
        batch (str): We update every batch evaluation
        epoch (str): We update every epoch
        valid (str): We update every validation
    """
    batch = 'batch'
    epoch = 'epoch'
    valid = 'valid'


StepIntervalUnitChoices = [step_interval.name for step_interval in StepIntervalUnit]
