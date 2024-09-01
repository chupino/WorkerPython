git clone https://github.com/chupino/WorkerPython.git worker
cd worker

docker build -t worker .

if [ $? -eq 0 ]; then
    echo "contruida"
else
    echo "mal"
    exit 1
fi

docker run -dp 5000:5000 worker