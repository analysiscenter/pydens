""" Contains specific Exceptions """


class BaseDatasetException(Exception):
    """ Base exception class """
    pass

class SkipBatchException(BaseDatasetException):
    """ Throw this in an action-method if you want to skip the batch from the rest of the pipeline """
    pass
