export PYSPARK_DRIVER_PYTHON=/home/odin/ryanwangdong_i/python2.7.14/bin/python
export PYSPARK_PYTHON=./python2.7.14/bin/python

/usr/local/spark-current/bin/spark-submit \
    --conf "spark.yarn.dist.archives=/home/odin/ryanwangdong_i/python2.7.14.tgz" \
    --conf "spark.scheduler.listenerbus.eventqueue.size=10000000" \
    --conf "spark.kryoserializer.buffer.max=2045m" \
    --conf "spark.kryoserializer.buffer=64m" \
    --driver-memory 30g \
    --executor-memory 12g \
    --queue celuemoxingbu_map_service \
    --num-executors 60 \
    pull_data.py 20180401 20180501 \
    | tee log.txt
    #--driver-java-options "-Xss100m" \
