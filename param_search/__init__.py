from .params import ParamSpace
from .job_scripts import setup_job_scripts as setup
from .job_queues import SlurmQueue, TorqueQueue
from .job_output import get_job_errors as errors
from .job_output import get_job_output as output

submit = SlurmQueue.submit_job_scripts
status = SlurmQueue.get_job_status
