from tuner import Tuner

job = Tuner(GPU=1)
job.manual_tune(2)
