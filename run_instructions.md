# Instructions

1. **Download the exam dataset**.

2. **Create and activate the virtual environment**:
    ```
    python -m venv automl-tabular-env
    source automl-tabular-env/bin/activate
    ```

    *(If you use a different environment name, update line 15 of the script accordingly.)*
    *(Make sure to set up the environment into the root directory of the project)*

3. **Install the dependencies**:
    ```
    pip install -r requirements.txt
    ```

4. **Update the script**:
   - Change the path on **line 14** of the script to point to your project's root directory.

5. **Run the pipeline script** from the project root:
    ```
    sbatch src/pipeline/final_pipeline.sh
    ```

The trained models and final predictions will be saved in the directory specified by `MODEL_DIR`.
