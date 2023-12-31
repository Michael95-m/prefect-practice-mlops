# prefect-practice-mlops
This is just a repo where I practiced using prefect in some use cases.


First, We need to upload the raw data into S3 bucket. So, let's create a s3 bucket to save the raw data and create a folder named `data` inside that bucket. Download the `diabetes_prediction_dataset.csv` from this [link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?select=diabetes_prediction_dataset.csv) and  upload this data into the `data folder` of the s3 bucket you created before. 

Let's export enviroment variable for `aws credentails` and `s3 bucket's name` that you save the data in your terminal.
```shell
export AWS_ACCESS_KEY_ID=****
export AWS_SECRET_ACCESS_KEY=****
export S3_BUCKET_NAME=**** ## bucket that you save the raw data for training pipeline
```
Replace **** with your credentials and your bucket name.

## Running Training pipeline

Before starting training pipeline, we have to deal with the blocks that can save aws credentials and s3 bucket. To save this credential, we need to run **create_s3_bucket_block.py**
```bash
python create_s3_bucket_block.py
```

To run the training pipeline, use the following command.
```bash
python training_pipeline.py
```

## Prefect deployment

In order to deploy our workflow, we need to initiate the prefect project. We can do this by using `prefect init`.  To me, I wanted to deploy the workflow by using github repository (This means the git repo will be cloned to the server and the workflow pipeline will be run). For this, extra parameter will be needed to be added.
```
prefect init --recipe git
```

There are two ways to create the deployment configuration. The first one is editing the deployment part of the **prefect.yaml**. In prefect.yaml,we can give the **name of the deployment**, **description**, **schedule**. We must also provide **entrypoint** which is the flow inside the python script. We can also give the parameters of the flow. We must also give **work_pool** name to run the workflow.

After that, we can deploy our workflow.
```
prefect deploy --name <deployment-name>
```

The second way is directly deploy our workflow directly without using **prefect.yaml**. We can do this by
```
prefect deploy <script.py>:<flow-name> -n <deployment-name> -p <pool-name> 
```

To create work-pool as a system process, use the following command. Replace <pool-name> with the name you like such as train-pool etc. This <pool-name> must be from the name we give in prefect.yaml. Actually, we can skip this steps because we can also create work-pool by using `prefect worker start` command.
```bash
prefect work-pool create --type process <pool-name>
```

Then we need to start our worker in work-pool to run the flow. If work-pool with <pool-name> didn't exist, that work-pool will be created.
```bash
prefect worker start --pool <pool-name>
```

After that, we can run our deployment flow. 
```bash
prefect deployment run <script-file-name>/<deployment-name>
```
The `<script-file-name>` is the name of the script file where our flow function exists and  `<deployment-name>` is the name of our deployment.