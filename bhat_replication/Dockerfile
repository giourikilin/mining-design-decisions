FROM python:3.8 AS py3
FROM openjdk:11
COPY --from=py3 / /
COPY requirements.txt /requirements.txt
RUN python -m pip install -r /requirements.txt
COPY classifiers.py /classifiers.py
CMD spark-submit --master $SPARK_MASTER_URL classifiers.py --algorithm x --file /detection_data_issues.json --test-file detection_data_ebse_issues_raw.json --split-size 0.9 --max-from-class 790 --benchmark 1