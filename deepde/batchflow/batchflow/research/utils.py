""" Auxilary functions """

def get_metrics(iteration, experiment, pipeline, metrics_var, metrics_name):
    """ Function to evaluate metrics """
    _ = iteration
    pipeline = experiment[pipeline].pipeline
    metrics = pipeline.get_variable(metrics_var)
    return metrics.evaluate(metrics_name)
