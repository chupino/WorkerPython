rm -rf worker
docker rm -f $(docker ps -lq)
docker rmi -f worker