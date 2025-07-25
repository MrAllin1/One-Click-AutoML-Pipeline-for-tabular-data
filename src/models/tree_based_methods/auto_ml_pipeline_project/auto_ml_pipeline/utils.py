"""
Utility functions: logging, GPU detection, persistence, reporting.
"""
import os
import json
import logging
import torch


def setup_logging(output_dir):
    logging.basicConfig(
        filename=os.path.join(output_dir,'pipeline.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('AutoML pipeline started')


def save_model(model,path):
    import joblib
    joblib.dump(model,path)


def report_results(metrics,output_dir):
    with open(os.path.join(output_dir,'metrics.json'),'w') as f:
        json.dump(metrics,f,indent=4)
