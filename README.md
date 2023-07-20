# prefect-practice-mlops
This is just a repo where I practiced using prefect in some use cases.


First, We need to upload the raw data into S3 bucket. So, let's create a s3 bucket to save the raw data and create a folder named `data` inside that bucket. Download the `diabetes_prediction_dataset.csv` from this [link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?select=diabetes_prediction_dataset.csv) and  upload this data into the `data folder` of the s3 bucket you created before. 

Let's export enviroment variable for `aws credentails` and `s3 bucket's name` that you save the data in your terminal.
```shell
export AWS_ACCESS_KEY_ID=****
export AWS_SECRET_ACCESS_KEY=****
export S3_BUCKET_NAME=**** ## bucket that you save the raw data for training pipeline
```

bash
```
python training_pipeline.py
```

```bash
prefect server start 

prefect work-pool create --type process train-pool

prefect worker start --pool train-pool

prefect deploy --name train-pipeline

 prefe

```