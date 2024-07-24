### Setup

Follow these steps to ensure you have all the required package to run this project.

- Create new Virtual Environment
```
    python3 -m venv vodth_ml_env
```

- Activate the Env
```
    source vodth_ml_env/bin/activate
```

- Install the required packages
```
    pip install -r requirements.txt
```

### Run the project

- To run this project run this command
```
    uvicorn app.main:app --host 0.0.0.0 --port 8000
```

