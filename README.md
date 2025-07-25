[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/sGN80kG1)
# AutoML Exam - SS25 (Tabular Data)
This repo serves as a template for the exam assignment of the AutoML SS25 course
at the university of Freiburg.

The aim of this repo is to provide a minimal installable template to help you set up and running.


## Installation

To install the repository, first create an environment of your choice and activate it. 


**Virtual Environment**

```bash
python3 -m venv automl-tabular-env
source automl-tabular-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-tabular-env python=3.11
conda activate automl-tabular-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:
```bash
python -c "import automl"
```

We place no restrictions on the Python version or libraries you use, but we recommend using Python 3.10 or higher.

## Code
We provide the following:

* `download-datasets.py`: This script downloads the suggested training datasets that we provide ahead of time, before the official exam dataset becomes available.

* `run.py`: A script that loads in a downloaded dataset, trains an _AutoML-System_ and then generates predictions for
`X_test`, saving those predictions to a file. For the training datasets, you will also have access to `y_test` which
is present in the `./data` folder, however you **will not** have access to `y_test` for the exam dataset.
Instead you will generate the predictions for `X_test` and submit those to us through Github Classroom.

* `./src/automl`: This is a python package that will be installed above and contain your source code for whatever
system you would like to build. We have provided a dummy `AutoML` class to serve as an example.

**You are completely free to modify, install new libraries, make changes and in general do whatever you want with the
code.** The only requirement for the exam will be that you can generate predictions for `X_test` in a `.npy` file
that we can then use to give you a test score through Github Classroom.

## Data

### Practice datasets:
The following datasets are provided for practice purposes:

* bike_sharing_demand
* brazilian_houses 
* wine_quality
* superconductivity 
* yprop_4_1

You can download the practice data using:
```bash
python download-datasets.py
```

This will by default, download the data to the `/data` folder with the following structure.
The fold numbers `1, ..., n` refer to **outer folds**, meaning each can be treated as a separate dataset for training and validation. You can use the `--fold` argument to specify which fold you would like.

```bash
./data
├── bike_sharing_demand
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 2
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 3
    ...
├── wine_quality 
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
    ...
```

## Running an initial test
This will train a dummy AutoML system and generate predictions for `X_test`:
```bash
python run.py --task bike_sharing_demand --seed 42 --output-path preds-42-bsd.npy
```

You are free to modify these files and command line arguments as you see fit.

## Exam

Download the exam/test dataset using:
```bash
python download-exam-dataset.py
```
The exam dataset has only one fold. You will need to generate predictions for this fold.
In the exam dataset, `y_test` is intentionally **not provided**.  

You must generate a predictions file named: `predictions.npy`  

- Save this `.npy` file in the folder **exam_dataset**.

Once you push your code, GitHub Actions will automatically detect this predictions file, compute the R² score for the exam dataset, and push the results back to your branch.

Running for exam_dataset:
```bash
python run.py --task exam_dataset --seed 42 --output-path data/exam_dataset/predictions.npy
```

## Running auto evaluation on test Dataset
Only activates on push to the test branch. It is important to note that Github Classroom creates unrelated histories for the main branch and test branch, that is why you can not use git merge main from the test branch directly. There are many ways to move the changes from other branches (e.g. from the main branch) to the test branch even though the commit histories between the branches are unrelated. Here is a simple way:

```bash
# on some_branch (e.g. main) do:
git add data/exam_dataset/predictions.npy
git commit -m "Generated predictions for test data"
git checkout test
#now you should be in the test branch
git checkout main -- data/exam_dataset/predictions.npy # only copies the data/exam_dataset/predictions.npy to the test branch and stages it, ready to be comitted
git status # ensure that your latest `.data/exam_dataset/predictions.npy` is staged
git commit -m "Generated predictions for test data, ready for evaluation"
git push
# wait for some time (few seconds) or monitor the web UI of Github to see if the job ran successfully
git pull 
# test scores will be downloaded under `.data/exam_dataset/test_out/` if the job ran successfully
```

Feel free to use any other command to move the prediction files from other branches with unrelated histories to the test branch (`rebase`,`merge some_branch_with_unrelated_history --allow-unrelated-histories`, `stash`...), **<span style="color:red">just make sure that there is nothing else inside `data/exam_dataset/` except for `predictions.npy` and the evaluation results that we push</span>**.

A summary of the evaluation workflow:
* To initialize auto-evaluation for the test data, checkout to the `test` branch.
* Make sure you have named the prediction file `predictions.npy` and placed it in the `data/exam_dataset/` directory in this branch.
* After pushing to it, the evaluation script will be automatically triggered.
* The results are also pushed to your repo (don't forget to `git pull`)
* If no new commits are pulled by `git pull`, check the errors in the Github's `Action` section (red cross inline, last commit message, test branch)

### <span style="color:red">Important: The dir `data/exam_dataset/` should contain only the `predictions.npy` file and the result files we push, nothing else; The exam_dataset you download should remain local and must not be pushed to the repository.</span>.

```bash
./data
└── exam_dataset
│   └── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_train.parquet
│   └── predictions.npy
│   └── test_out
│   │   ├── test_evaluation_output_2025-MM-DD_HH-mm-ss-ms
.   .   .
```

<span style="color:red"> **Note that any edits to the yaml workflow script are prohibited and monitored!** </span>


## Reference performance

| Dataset | Test performance |
| -- | -- |
| bike_sharing_demand | 0.9457 |
| brazilian_houses | 0.9896 |
| superconductivity | 0.9311 |
| wine_quality | 0.4410 |
| yprop_4_1 | 0.0778 |
| exam_dataset | 0.9290 |

The scores listed are the R² values calculated using scikit-learn's `metrics.r2_score`.

## Final submission

The following must be submitted by `August 6, 2025, 23:59 CET` for a successful project submission and poster participation:

#### **1) Poster submission**
Upload your poster as a PDF file named as `final_poster_tabular_<team-name>.pdf`, following the template given [here](https://docs.google.com/presentation/d/1T55GFGsoon9a4T_oUm4WXOhW8wMEQL3M/edit?slide=id.p1#slide=id.p1).

#### **2) Test predictions**
The final test predictions should be uploaded in a file `predictions.npy`, with each line containing the predictions for the input in the exact order of `X_test` given.

#### **3) Reproducibility instructions**
TL;DR: Code and instructions to _reproduce_ the above test predictions.

A `run_instructions.md` file that guides through the command to run the designed AutoML solution on the training set of the *final-test-dataset*.
This command should return either a: (i) hyperparameter configuration, (ii) a partially trained model on a hyperparameter configuration, or (iii) a fully trained model in `24 hours` at most.
A second command that given (i), (ii), or (iii) would do the needful that yields predictions for `test_X` for the *final-test-dataset*. This is the `predictions.npy`.

#### **4) Team information**
Upload a file `team_info.txt` with the list of matriculation IDs of team members (*NO NAMES*). (E.g.: 1234567, 7654321)

### Submission checklist:
- [ ] Poster
- [ ] Test predictions
- [ ] Reproducibility instructions
- [ ] Team info
- [x] *Example to denote task being done*
<!-- This is a comment. -->


## Tips
* If you need to add dependencies that you and your teammates are all on the same page, you can modify the
`pyproject.toml` file and add the dependencies there. This will ensure that everyone has the same dependencies

* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
predictions, etc. Also, be a friendly teammate and ignore your virtual environment and any additional folders/files
created by your IDE.